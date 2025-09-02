// ============================================================================
//  Worker ArmoniK — Cholesky par blocs (POTRF, TRSM, SYRK, GEMM)
//  Version simple : logs stdout "logger.info(...)" avant / après les appels
//  (on n'ajoute rien autour des sections chronométrées)
// ============================================================================

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cmath>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include <fstream>
#include <stdexcept>
#include <iterator>

#include <grpcpp/grpcpp.h>
#include "grpcpp/support/sync_stream.h"

#include "armonik/common/objects.pb.h"
#include "armonik/common/exceptions/ArmoniKApiException.h"

#include "armonik/worker/utils/WorkerServer.h"
#include "armonik/worker/Worker/ArmoniKWorker.h"
#include "armonik/worker/Worker/ProcessStatus.h"
#include "armonik/worker/Worker/TaskHandler.h"

#include <chameleon.h>

// ============================================================================
// Utilitaires
// ============================================================================

static int env_int(const char* key, int defval) {
  if (const char* s = std::getenv(key)) {
    try { return std::max(0, std::stoi(s)); } catch (...) {}
  }
  return defval;
}

static std::map<std::string,std::string> parse_kv_payload(const std::string& s) {
  std::map<std::string,std::string> kv;
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) {
    auto pos = tok.find('=');
    if (pos != std::string::npos) kv[tok.substr(0,pos)] = tok.substr(pos+1);
  }
  return kv;
}

static int to_int(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return std::stoi(it->second);
}

static std::string to_str(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return it->second;
}

static std::vector<double> deserialize_block(const std::string& blob, int B) {
  const size_t need = static_cast<size_t>(B) * static_cast<size_t>(B) * sizeof(double);
  if (blob.size() != need)
    throw std::runtime_error("Invalid block size: expected " + std::to_string(need) +
                              " bytes, got " + std::to_string(blob.size()));
  std::vector<double> v(B*B);
  std::memcpy(v.data(), blob.data(), need);
  return v;
}

static std::string serialize_block(const std::vector<double>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

// Télécharge un blob d'entrée depuis les dépendances data (IDs -> chemin)
static std::string download_blob(armonik::api::worker::TaskHandler& th,
  const std::string& resultId)
{
  const auto& deps = th.getDataDependencies();
  auto it = deps.find(resultId);
  if (it == deps.end()) {
    throw std::runtime_error("Input '" + resultId + "' not found in data dependencies");
  }
  const std::string& path = it->second;
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open dependency file: " + path);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static void upload_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId, const std::string& data) {
  th.send_result(resultId, data).get();
}

static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
}

// ============================================================================
// Worker
// ============================================================================

class DagCholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent)
  : ArmoniKWorker(std::move(agent)) {}

  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    try {
      // getPayload
      std::cout << "logger.info(\"before getPayload\")" << std::endl;
      const std::string payload = taskHandler.getPayload();
      std::cout << "logger.info(\"success getPayload\")" << std::endl;

      // parse_kv_payload
      std::cout << "logger.info(\"before parse_kv_payload\")" << std::endl;
      auto kv = parse_kv_payload(payload);
      std::cout << "logger.info(\"success parse_kv_payload\")" << std::endl;

      // to_str / to_int
      std::cout << "logger.info(\"before to_str(op)\")" << std::endl;
      const std::string op = to_str(kv, "op");
      std::cout << "logger.info(\"success to_str(op)\")" << std::endl;

      std::cout << "logger.info(\"before to_int(B)\")" << std::endl;
      const int B = to_int(kv, "B");
      std::cout << "logger.info(\"success to_int(B)\")" << std::endl;

      // IDs in/out selon op
      std::string in, inL, inA, inC, inAi, inAj, out;
      if (op == "POTRF") {
        std::cout << "logger.info(\"before to_str(in)\")" << std::endl;
        in = to_str(kv, "in");
        std::cout << "logger.info(\"success to_str(in)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(out)\")" << std::endl;
        out = to_str(kv, "out");
        std::cout << "logger.info(\"success to_str(out)\")" << std::endl;
      }
      else if (op == "TRSM") {
        std::cout << "logger.info(\"before to_str(inL)\")" << std::endl;
        inL = to_str(kv, "inL");
        std::cout << "logger.info(\"success to_str(inL)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(inA)\")" << std::endl;
        inA = to_str(kv, "inA");
        std::cout << "logger.info(\"success to_str(inA)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(out)\")" << std::endl;
        out = to_str(kv, "out");
        std::cout << "logger.info(\"success to_str(out)\")" << std::endl;
      }
      else if (op == "SYRK") {
        std::cout << "logger.info(\"before to_str(inC)\")" << std::endl;
        inC = to_str(kv, "inC");
        std::cout << "logger.info(\"success to_str(inC)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(inA)\")" << std::endl;
        inA = to_str(kv, "inA");
        std::cout << "logger.info(\"success to_str(inA)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(out)\")" << std::endl;
        out = to_str(kv, "out");
        std::cout << "logger.info(\"success to_str(out)\")" << std::endl;
      }
      else if (op == "GEMM") {
        std::cout << "logger.info(\"before to_str(inC)\")" << std::endl;
        inC = to_str(kv, "inC");
        std::cout << "logger.info(\"success to_str(inC)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(inAi)\")" << std::endl;
        inAi = to_str(kv, "inAi");
        std::cout << "logger.info(\"success to_str(inAi)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(inAj)\")" << std::endl;
        inAj = to_str(kv, "inAj");
        std::cout << "logger.info(\"success to_str(inAj)\")" << std::endl;

        std::cout << "logger.info(\"before to_str(out)\")" << std::endl;
        out = to_str(kv, "out");
        std::cout << "logger.info(\"success to_str(out)\")" << std::endl;
      }
      else {
        return armonik::api::worker::ProcessStatus("Unknown op=" + op);
      }

      // env + init CHAMELEON
      std::cout << "logger.info(\"before env_int(CHM_NCPU)\")" << std::endl;
      int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency()));
      std::cout << "logger.info(\"success env_int(CHM_NCPU)\")" << std::endl;

      std::cout << "logger.info(\"before env_int(CHM_NGPU)\")" << std::endl;
      int ngpu = env_int("CHM_NGPU", 0);
      std::cout << "logger.info(\"success env_int(CHM_NGPU)\")" << std::endl;

      std::cout << "logger.info(\"before CHAMELEON_Init\")" << std::endl;
      CHAMELEON_Init(ncpu, ngpu);
      std::cout << "logger.info(\"success CHAMELEON_Init\")" << std::endl;

      int info = 0;
      std::vector<double> A(B*B), L(B*B), C(B*B), Ai(B*B), Aj(B*B);
      CHAM_desc_t *dA=nullptr, *dL=nullptr, *dC=nullptr, *dAi=nullptr, *dAj=nullptr;

      // Chargements / desc
      if (op == "POTRF") {
        std::cout << "logger.info(\"before download_blob(in)\")" << std::endl;
        auto blob = download_blob(taskHandler, in);
        std::cout << "logger.info(\"success download_blob(in)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(A)\")" << std::endl;
        A = deserialize_block(blob, B);
        std::cout << "logger.info(\"success deserialize_block(A)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dA)\")" << std::endl;
        create_desc_1block(&dA, A.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dA)\")" << std::endl;
      } else if (op == "TRSM") {
        std::cout << "logger.info(\"before download_blob(inL)\")" << std::endl;
        auto blobL = download_blob(taskHandler, inL);
        std::cout << "logger.info(\"success download_blob(inL)\")" << std::endl;

        std::cout << "logger.info(\"before download_blob(inA)\")" << std::endl;
        auto blobA = download_blob(taskHandler, inA);
        std::cout << "logger.info(\"success download_blob(inA)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(L)\")" << std::endl;
        L = deserialize_block(blobL, B);
        std::cout << "logger.info(\"success deserialize_block(L)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(A)\")" << std::endl;
        A = deserialize_block(blobA, B);
        std::cout << "logger.info(\"success deserialize_block(A)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dL)\")" << std::endl;
        create_desc_1block(&dL, L.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dL)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dA)\")" << std::endl;
        create_desc_1block(&dA, A.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dA)\")" << std::endl;
      } else if (op == "SYRK") {
        std::cout << "logger.info(\"before download_blob(inC)\")" << std::endl;
        auto blobC = download_blob(taskHandler, inC);
        std::cout << "logger.info(\"success download_blob(inC)\")" << std::endl;

        std::cout << "logger.info(\"before download_blob(inA)\")" << std::endl;
        auto blobA = download_blob(taskHandler, inA);
        std::cout << "logger.info(\"success download_blob(inA)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(C)\")" << std::endl;
        C = deserialize_block(blobC, B);
        std::cout << "logger.info(\"success deserialize_block(C)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(A)\")" << std::endl;
        A = deserialize_block(blobA, B);
        std::cout << "logger.info(\"success deserialize_block(A)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dC)\")" << std::endl;
        create_desc_1block(&dC, C.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dC)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dA)\")" << std::endl;
        create_desc_1block(&dA, A.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dA)\")" << std::endl;
      } else { // GEMM
        std::cout << "logger.info(\"before download_blob(inC)\")" << std::endl;
        auto blobC = download_blob(taskHandler, inC);
        std::cout << "logger.info(\"success download_blob(inC)\")" << std::endl;

        std::cout << "logger.info(\"before download_blob(inAi)\")" << std::endl;
        auto blobAi = download_blob(taskHandler, inAi);
        std::cout << "logger.info(\"success download_blob(inAi)\")" << std::endl;

        std::cout << "logger.info(\"before download_blob(inAj)\")" << std::endl;
        auto blobAj = download_blob(taskHandler, inAj);
        std::cout << "logger.info(\"success download_blob(inAj)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(C)\")" << std::endl;
        C = deserialize_block(blobC, B);
        std::cout << "logger.info(\"success deserialize_block(C)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(Ai)\")" << std::endl;
        Ai = deserialize_block(blobAi, B);
        std::cout << "logger.info(\"success deserialize_block(Ai)\")" << std::endl;

        std::cout << "logger.info(\"before deserialize_block(Aj)\")" << std::endl;
        Aj = deserialize_block(blobAj, B);
        std::cout << "logger.info(\"success deserialize_block(Aj)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dC)\")" << std::endl;
        create_desc_1block(&dC, C.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dC)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dAi)\")" << std::endl;
        create_desc_1block(&dAi, Ai.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dAi)\")" << std::endl;

        std::cout << "logger.info(\"before create_desc_1block(dAj)\")" << std::endl;
        create_desc_1block(&dAj, Aj.data(), B);
        std::cout << "logger.info(\"success create_desc_1block(dAj)\")" << std::endl;
      }

      // Chronométrage + calcul : on NE LOG PAS avant/après les appels CHAMELEON_* ici
      struct timespec t0{}, t1{};
      auto Bd = static_cast<double>(B);
      double secs = 0.0, flops = 0.0, gflops = 0.0;

      if (op == "POTRF") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) throw std::runtime_error("dpotrf info=" + std::to_string(info));
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        std::cout << "logger.info(\"before upload_blob(out)\")" << std::endl;
        upload_blob(taskHandler, out, serialize_block(A));
        std::cout << "logger.info(\"success upload_blob(out)\")" << std::endl;
      }
      else if (op == "TRSM") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 0.5 * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        std::cout << "logger.info(\"before upload_blob(out)\")" << std::endl;
        upload_blob(taskHandler, out, serialize_block(A));
        std::cout << "logger.info(\"success upload_blob(out)\")" << std::endl;
      }
      else if (op == "SYRK") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        std::cout << "logger.info(\"before upload_blob(out)\")" << std::endl;
        upload_blob(taskHandler, out, serialize_block(C));
        std::cout << "logger.info(\"success upload_blob(out)\")" << std::endl;
      }
      else { // GEMM
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 2.0 * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        std::cout << "logger.info(\"before upload_blob(out)\")" << std::endl;
        upload_blob(taskHandler, out, serialize_block(C));
        std::cout << "logger.info(\"success upload_blob(out)\")" << std::endl;
      }

      // Destruction descripteurs
      std::cout << "logger.info(\"before destroy descriptors\")" << std::endl;
      if (dA) CHAMELEON_Desc_Destroy(&dA);
      if (dL) CHAMELEON_Desc_Destroy(&dL);
      if (dC) CHAMELEON_Desc_Destroy(&dC);
      if (dAi) CHAMELEON_Desc_Destroy(&dAi);
      if (dAj) CHAMELEON_Desc_Destroy(&dAj);
      std::cout << "logger.info(\"success destroy descriptors\")" << std::endl;

      // Finalize
      std::cout << "logger.info(\"before CHAMELEON_Finalize\")" << std::endl;
      CHAMELEON_Finalize();
      std::cout << "logger.info(\"success CHAMELEON_Finalize\")" << std::endl;

      return armonik::api::worker::ProcessStatus::Ok;
    }
    catch (const std::exception& e) {
      std::cerr << "logger.info(\"FAIL in Execute : " << e.what() << "\")" << std::endl;
      std::cerr << "[DagWorker] ERROR: " << e.what() << "\n";
      return armonik::api::worker::ProcessStatus(e.what());
    }
  }
};

int main() {
  // Flush plus agressif pour voir tous les logs
  std::ios::sync_with_stdio(false);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  std::cout << "DagCholeskyWorker started. gRPC version = " << grpc::Version() << std::endl;

  armonik::api::common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  config.set("ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock");
  config.set("ComputePlane__AgentChannel__Address", "/cache/armonik_agent.sock");

  try {
    armonik::api::worker::WorkerServer::create<DagCholeskyWorker>(config)->run();
  } catch (const std::exception &e) {
    std::cerr << "Error in worker: " << e.what() << std::endl;
  }

  std::cout << "Stopping Server..." << std::endl;
  return 0;
}
