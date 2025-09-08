


/////////////////////////////////////////
// ============================================================================
//  Worker ArmoniK — Cholesky par blocs 
//  POTRF, TRSM, SYRK, GEMM sur un bloc BxB
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
#include <regex>
#include <algorithm>
#include <filesystem>



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

// Nettoie un ID pour éviter les caractères bizarres
static std::string clean_id(std::string s) {
  s = std::regex_replace(s, std::regex("[^0-9a-fA-F\\-]"), "");
  return s;
}


// N’accepte qu’un UUID canonique (prend le 1er match)
static std::string canonical_uuid(const std::string& s) {
  std::smatch m;
  if (std::regex_search(s, m, std::regex(
      R"(([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}))")))
    return m[1].str();
  throw std::runtime_error("Bad UUID in payload field: " + s.substr(0,64));
}

static std::string sanitize_utf8_ascii(std::string s) {
  std::string out; out.reserve(s.size());
  for (unsigned char c : s) {
    if (c == '\n' || c == '\r' || (c >= 32 && c < 127)) out.push_back((char)c);
    else out.push_back('?');
  }
  return out;
}

// --- DIAG: hexdump pour voir si le payload contient du binaire
static void hexdump(const std::string& s, size_t max=256) {
  std::cerr << "[payload] size=" << s.size() << "\n[payload] hex:";
  for (size_t i=0; i<std::min(max, s.size()); ++i) {
    unsigned char c = (unsigned char)s[i];
    if (i%16==0) std::cerr << "\n  ";
    static const char* h="0123456789ABCDEF";
    std::cerr << h[c>>4] << h[c&0xF] << ' ';
  }
  std::cerr << "\n";
}

// --- Ne garde que la toute première ligne/partie texte du payload
static std::string header_from_payload(const std::string& payload) {
  size_t cut = payload.size();
  size_t n0  = payload.find('\0');
  size_t nl  = payload.find('\n');
  if (n0 != std::string::npos) cut = std::min(cut, n0);
  if (nl != std::string::npos) cut = std::min(cut, nl);
  cut = std::min(cut, (size_t)1024); // garde-fou
  return payload.substr(0, cut);
}









// Télécharge un blob d'entrée depuis les dépendances data (IDs -> chemin)
static std::string download_blob(armonik::api::worker::TaskHandler& th,
                                 const std::string& resultId)
{
  const auto& deps = th.getDataDependencies();
  auto it = deps.find(resultId);
  if (it == deps.end()) {
    // Log clair si l’ID n’est pas déclaré dans les deps
    std::ostringstream oss;
    oss << "Input id not found in dependencies: " << resultId
        << " (known ids=";
    bool first=true;
    for (auto &kv : deps) { if(!first) oss<<","; first=false; oss<<kv.first; }
    oss << ")";
    throw std::runtime_error(oss.str());
  }

  const std::string& path = it->second;  // <<< IMPORTANT
  std::cerr << "[DagWorker] opening dep id=" << resultId << " path=" << path << "\n";

  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    throw std::runtime_error("Dependency file missing at path: " + path +
                             " (id=" + resultId + (ec ? ", fs_error=" + ec.message() : "") + ")");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.good()) {
    std::ostringstream oss;
    oss << "Cannot open dependency file: " << path
        << " (id=" << resultId << ", rdstate=0x" << std::hex << f.rdstate() << ")";
    throw std::runtime_error(oss.str());
  }

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

      // Log pour voir le mapping
      const auto& deps = taskHandler.getDataDependencies();
      std::cerr << "[DagWorker] Deps count=" << deps.size() << "\n";
      for (auto& kvp : deps) {
        std::cerr << "  depId=" << kvp.first << " path=" << kvp.second << "\n";
        }

      const std::string payload = taskHandler.getPayload();
        
      hexdump(payload);
      // On ne parse que l’en-tête texte
      const std::string header = header_from_payload(payload);
      auto kv = parse_kv_payload(header);
      //auto kv = parse_kv_payload(payload);
      
      const std::string op = to_str(kv, "op");
      const int B = to_int(kv, "B");

      // Récupération des IDs d'entrée/sortie depuis le payload
      std::string in, inL, inA, inC, inAi, inAj, out;
      if (op == "POTRF") { in = to_str(kv, "in"); out = to_str(kv, "out"); }
      else if (op == "TRSM") { inL = to_str(kv, "inL"); inA = to_str(kv, "inA"); out = to_str(kv, "out"); }
      else if (op == "SYRK") { inC = to_str(kv, "inC"); inA = to_str(kv, "inA"); out = to_str(kv, "out"); }
      else if (op == "GEMM") { inC = to_str(kv, "inC"); inAi= to_str(kv, "inAi"); inAj= to_str(kv, "inAj"); out= to_str(kv, "out"); }
      else return armonik::api::worker::ProcessStatus("Unknown op=" + op);

      // 4) Nettoyage des IDs (après lecture)
      in   = clean_id(in);
      inL  = clean_id(inL);
      inA  = clean_id(inA);
      inC  = clean_id(inC);
      inAi = clean_id(inAi);
      inAj = clean_id(inAj);
      out  = clean_id(out);

      // Eviter d'avoir plusieurs initialisition si plusieurs execute
      //int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency()));
      //int ngpu = env_int("CHM_NGPU", 0);
      //CHAMELEON_Init(ncpu, ngpu);

      // Nettoyage + canonicalisation des IDs
      auto canon = [](const std::string& s)->std::string {
        if (s.empty()) return s;
        return canonical_uuid(clean_id(s));
      };
      if (!in.empty())   in   = canon(in);
      if (!inL.empty())  inL  = canon(inL);
      if (!inA.empty())  inA  = canon(inA);
      if (!inC.empty())  inC  = canon(inC);
      if (!inAi.empty()) inAi = canon(inAi);
      if (!inAj.empty()) inAj = canon(inAj);
      if (!out.empty())  out  = canon(out);

      
      std::cerr << "[DagWorker] header=\"" << sanitize_utf8_ascii(header) << "\"\n";
      std::cerr << "[DagWorker] op="<<op<<" B="<<B
                << " in="<<in<<" inL="<<inL<<" inA="<<inA
                << " inC="<<inC<<" inAi="<<inAi<<" inAj="<<inAj
                << " out="<<out << "\n";

      int info = 0;
      std::vector<double> A(B*B), L(B*B), C(B*B), Ai(B*B), Aj(B*B);
      CHAM_desc_t *dA=nullptr, *dL=nullptr, *dC=nullptr, *dAi=nullptr, *dAj=nullptr;


      if (op == "POTRF") {
        A = deserialize_block(download_blob(taskHandler, in), B);
        create_desc_1block(&dA, A.data(), B);
      } else if (op == "TRSM") {
        L = deserialize_block(download_blob(taskHandler, inL), B);
        A = deserialize_block(download_blob(taskHandler, inA), B);
        create_desc_1block(&dL, L.data(), B);
        create_desc_1block(&dA, A.data(), B);
      } else if (op == "SYRK") {
        C = deserialize_block(download_blob(taskHandler, inC), B);
        A = deserialize_block(download_blob(taskHandler, inA), B);
        create_desc_1block(&dC, C.data(), B);
        create_desc_1block(&dA, A.data(), B);
      } else { // GEMM
        C = deserialize_block(download_blob(taskHandler, inC), B);
        Ai = deserialize_block(download_blob(taskHandler, inAi), B);
        Aj = deserialize_block(download_blob(taskHandler, inAj), B);
        create_desc_1block(&dC, C.data(), B);
        create_desc_1block(&dAi, Ai.data(), B);
        create_desc_1block(&dAj, Aj.data(), B);
      }

      struct timespec t0{}, t1{};
      auto Bd = static_cast<double>(B);
      double secs = 0.0, flops = 0.0, gflops = 0.0;


      if (op == "POTRF") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) throw std::runtime_error("dpotrf info=" + std::to_string(info));
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * Bd*Bd*Bd; (void)gflops; gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(A));
      }
      else if (op == "TRSM") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 0.5 * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(A));
      }
      else if (op == "SYRK") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(C));
      }

      else { // GEMM
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 2.0 * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(C));
      }


      if (dA) CHAMELEON_Desc_Destroy(&dA);
      if (dL) CHAMELEON_Desc_Destroy(&dL);
      if (dC) CHAMELEON_Desc_Destroy(&dC);
      if (dAi) CHAMELEON_Desc_Destroy(&dAi);
      if (dAj) CHAMELEON_Desc_Destroy(&dAj);
      
      // CHAMELEON_Finalize();


      return armonik::api::worker::ProcessStatus::Ok;
    }
    catch (const std::exception& e) {

      // Ajout de la sanitise
      std::string msg = sanitize_utf8_ascii(e.what());

      std::cerr << "[DagWorker] ERROR: " << msg << "\n";
      return armonik::api::worker::ProcessStatus(msg);
    }
  }
};

int main() {
  std::cout << "DagCholeskyWorker started. gRPC version = " << grpc::Version() << "\n";
  
  // Init CHAMELEON une seule fois pour tout le process
  int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency()));
  int ngpu = env_int("CHM_NGPU", 0);
  CHAMELEON_Init(ncpu, ngpu);
  std::atexit([]{ CHAMELEON_Finalize(); });
  
  
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