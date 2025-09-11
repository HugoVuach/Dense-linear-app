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
#include <cerrno>
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
#include "armonik/worker/Worker/ProcessStatus.h"
#include "armonik/worker/Worker/TaskHandler.h"
#include "armonik/worker/Worker/ArmoniKWorker.h"
#include "armonik/worker/utils/WorkerServer.h"



#include <chameleon.h>   

// ============================================================================
// Serialize and Deserialize function
// ============================================================================

// 1. Deserialize function
static std::vector<double> deserialize_block(const std::string& blob, int B) {
  const size_t need = static_cast<size_t>(B) * static_cast<size_t>(B) * sizeof(double);
  if (blob.size() != need)
    throw std::runtime_error("Invalid block size: expected " + std::to_string(need) + " bytes, got " + std::to_string(blob.size()));
  std::vector<double> v(B*B);
  std::memcpy(v.data(), blob.data(), need);
  return v;
}

// 2. Serialize function
static std::string serialize_block(const std::vector<double>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

// ===========================================================================
// Download/Upload blob via TaskHandler using data dependencies
// ===========================================================================

// 1. Download blob function
static std::string download_blob(armonik::api::worker::TaskHandler& taskHandler, const std::string& resultId){
  taskHandler.getPayload().data();
  const auto& deps = taskHandler.getDataDependencies();
  auto it = deps.find(resultId);
  if (it == deps.end()){
    throw std::runtime_error("Dependency not provided: " + resultId);
  }
  const std::string& path = it->second;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    int e = errno;
    throw std::runtime_error("Cannot open dependency file " + path + " (id=" + resultId + ") errno=" + std::to_string(e) + " " + std::string(std::strerror(e)));
  }
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// 2. Upload blob function
static void upload_blob(armonik::api::worker::TaskHandler& taskHandler,
                        const std::string& resultId,
                        const std::string& data)
{
  if (resultId.empty())
    throw std::runtime_error("Empty resultId in upload_blob()");

  taskHandler.send_result(resultId, data).get();
}

// ==========================================================================
// Utils for Payload parsing and check format
// ==========================================================================

// 1. Parse Payload and check uuid format
static std::map<std::string,std::string> parse_kv_to_map(const std::string& s) {
  constexpr size_t MAX_TOKEN_LEN = 40;
  std::map<std::string,std::string> kv;

  for (size_t i = 0; i < s.size();) {
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    if (i >= s.size()) break;

    size_t j = i;
    while (j < s.size() && !std::isspace((unsigned char)s[j])) ++j;

    std::string tok = s.substr(i, j - i);
    size_t eq = tok.find('=');

    if (eq != std::string::npos && eq > 0 && eq + 1 < tok.size()) {
      std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
      size_t maxv = (MAX_TOKEN_LEN > k.size() + 1) ? MAX_TOKEN_LEN - (k.size() + 1) : 0;
      if (maxv && v.size() > maxv) v.resize(maxv);
      if (!v.empty()) kv[std::move(k)] = std::move(v);
    }
    i = j;
  }
  return kv;
}



// 2. Check if a string is a valid UUID format
static bool is_uuid36(const std::string& s){
  static const std::regex re(
    R"(^[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}$)"
  );
  return std::regex_match(s, re);
}


// 3. Check if a string is present in the map
static std::string need(const std::map<std::string,std::string>& kv, const char* key){
  auto it=kv.find(key); if(it==kv.end()||it->second.empty())
    throw std::runtime_error(std::string("Missing key '")+key+"'");
  return it->second;
}

// 4. Check if a string is a valid UUID
static std::string need_uuid(const std::map<std::string,std::string>& kv, const char* key){
  auto v = need(kv,key); if(!is_uuid36(v)) throw std::runtime_error(std::string("Invalid UUID for '")+key+"'");
  return v;
}

// 5. To get optional key
static std::string to_upper(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::toupper(c);});
  return s;
}

// 6. To get B
static int need_int(const std::map<std::string,std::string>& kv, const char* key){
  try { return std::stoi(need(kv,key)); } catch(...) { throw std::runtime_error(std::string("Bad integer for '")+key+"'"); }
}

// 6. Check if a string is a valid UUID format
static std::string sanitize_ascii(std::string s){
  std::string out; out.reserve(s.size());
  for (unsigned char c: s) out.push_back((c=='\n'||c=='\r'||(c>=32&&c<127))?(char)c:'?');
  return out;
}

// ==========================================================================
// Utils for Chameleon functions
// ==========================================================================

//1. Create desc for 1 block
static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
}

// 2. To get environnement variable for chameleon
static int env_int(const char* key, int defval) {
  if (const char* s = std::getenv(key)) { try { return std::max(0, std::stoi(s)); } catch (...) {} }
  return defval;
}


///////////////////////////////////////////////////////////////////////////////

// ============================================================================
// Worker
// ============================================================================


class DagCholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent)
  : ArmoniKWorker(std::move(agent)) {}


  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    try {
      std::cout << "[DagCholeskyWorker_try][INFO 1] ---- Execute begin ----\n";

      std::cout << "[DagCholeskyWorker_try][INFO 2] SizePayload : " << taskHandler.getPayload().size()
                << "\n [DagCholeskyWorker_try][INFO 3] Size DD : " << taskHandler.getDataDependencies().size()
                << "\n [DagCholeskyWorker_try][INFO 4] Expected results : " << taskHandler.getExpectedResults().size() << std::endl;

      const std::string payload = taskHandler.getPayload();
      const auto kv = parse_kv_to_map(payload);

      std::string op = to_upper(need(kv, "op"));
      int B = need_int(kv, "B");

      double secs = 0.0, flops = 0.0, gflops = 0.0;
      struct timespec t0{}, t1{};

      // ============================
      // POTRF
      // ============================
      if (op == "POTRF") {

        const std::string in_id  = need_uuid(kv, "in");
        const std::string out_id = need_uuid(kv, "out");

        const auto& expected = taskHandler.getExpectedResults();

        const std::string Ablob = download_blob(taskHandler, in_id);
        std::vector<double> A   = deserialize_block(Ablob, B);

        CHAM_desc_t *dA = nullptr;
        create_desc_1block(&dA, A.data(), B);
        int info = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) {CHAMELEON_Desc_Destroy(&dA);
          throw std::runtime_error("dpotrf info=" + std::to_string(info));}
        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        upload_blob(taskHandler, out_id, serialize_block(A));
        CHAMELEON_Desc_Destroy(&dA);
      }

      // ============================
      // TRSM
      // ============================
      else if (op == "TRSM") {
        const std::string inL    = need_uuid(kv, "inL");
        const std::string inA    = need_uuid(kv, "inA");
        const std::string out_id = need_uuid(kv, "out");

        const std::string Lblob = download_blob(taskHandler, inL);
        const std::string Ablob = download_blob(taskHandler, inA);
        std::vector<double> L = deserialize_block(Lblob, B);
        std::vector<double> A = deserialize_block(Ablob, B);

        CHAM_desc_t *dL = nullptr, *dA = nullptr;
        create_desc_1block(&dL, L.data(), B);
        create_desc_1block(&dA, A.data(), B);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs   = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops  = 0.5 * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        upload_blob(taskHandler, out_id, serialize_block(A));

        CHAMELEON_Desc_Destroy(&dL);
        CHAMELEON_Desc_Destroy(&dA);
      }

      // ============================
      // SYRK
      // ============================
      else if (op == "SYRK") {
        const std::string inC    = need_uuid(kv, "inC");
        const std::string inA    = need_uuid(kv, "inA");
        const std::string out_id = need_uuid(kv, "out");

        const std::string Cblob = download_blob(taskHandler, inC);
        const std::string Ablob = download_blob(taskHandler, inA);
        std::vector<double> C = deserialize_block(Cblob, B);
        std::vector<double> A = deserialize_block(Ablob, B);

        CHAM_desc_t *dC = nullptr, *dA = nullptr;
        create_desc_1block(&dC, C.data(), B);
        create_desc_1block(&dA, A.data(), B);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs   = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops  = 1.0 * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        upload_blob(taskHandler, out_id, serialize_block(C));

        CHAMELEON_Desc_Destroy(&dC);
        CHAMELEON_Desc_Destroy(&dA);
      }

      // ============================
      // GEMM
      // ============================
      else if (op == "GEMM") {
        const std::string inC    = need_uuid(kv, "inC");
        const std::string inAi   = need_uuid(kv, "inAi");
        const std::string inAj   = need_uuid(kv, "inAj");
        const std::string out_id = need_uuid(kv, "out");

        const std::string Cblob  = download_blob(taskHandler, inC);
        const std::string Aiblob = download_blob(taskHandler, inAi);
        const std::string Ajblob = download_blob(taskHandler, inAj);

        std::vector<double> C  = deserialize_block(Cblob,  B);
        std::vector<double> Ai = deserialize_block(Aiblob, B);
        std::vector<double> Aj = deserialize_block(Ajblob, B);
        CHAM_desc_t *dC = nullptr, *dAi = nullptr, *dAj = nullptr;
        create_desc_1block(&dC,  C.data(),  B);
        create_desc_1block(&dAi, Ai.data(), B);
        create_desc_1block(&dAj, Aj.data(), B);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs   = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops  = 2.0 * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        upload_blob(taskHandler, out_id, serialize_block(C));

        CHAMELEON_Desc_Destroy(&dC);
        CHAMELEON_Desc_Destroy(&dAi);
        CHAMELEON_Desc_Destroy(&dAj);
      }
      else {
        return armonik::api::worker::ProcessStatus("Unknown op=" + op);
      }

      std::cerr << "[INFO] Execute: end OK\n";
      std::cerr << "[PERF] " 
                << " time=" << secs << " sec"
                << " flops=" << flops << " gflops=" << gflops << "\n";

      return armonik::api::worker::ProcessStatus::Ok;
    } catch (const std::exception &e) {
      return armonik::api::worker::ProcessStatus(sanitize_ascii(e.what()));
    }
  }
};

int main() {  

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
  }
  return 0;
}


