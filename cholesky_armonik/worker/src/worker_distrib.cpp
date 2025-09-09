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


//static std::map<std::string,std::string> parse_kv_payload(const std::string& s) {
  //std::map<std::string,std::string> kv;
  //std::istringstream iss(s);
  //std::string tok;
  //while (iss >> tok) {
    //auto pos = tok.find('=');
    //if (pos != std::string::npos) kv[tok.substr(0,pos)] = tok.substr(pos+1);
  //}
  //return kv;
//}

struct ParsedPayload {
  std::string op;
  int         B = 0;
  std::string in, inL, inA, inC, inAi, inAj, out; // UUIDs (36 chars) ou vide si non fourni
};

static ParsedPayload parse_kv_payload(const std::string& payload) {
  ParsedPayload r;

  // --- helpers locaux (pas de fonctions globales supplémentaires) ---
  auto find_segment = [&](const std::string& key) -> std::string {
    // Cherche " key=" ou "key=" (début ou séparé par espace)
    size_t pos = payload.find(key + "=");
    if (pos == std::string::npos) {
      // tolère aussi " key=" avec espace avant
      std::string pat = " " + key + "=";
      pos = payload.find(pat);
      if (pos == std::string::npos) return {};
      pos += 1; 
    }
    size_t start = pos + key.size() + 1; 
    size_t i = start;
    while (i < payload.size()) {
      unsigned char c = (unsigned char)payload[i];
      if (c == 0 || c == '\n' || c == '\r' || std::isspace(c)) break;
      ++i;
    }
    return payload.substr(start, i - start);
  };

  auto first_uuid = [&](const std::string& s) -> std::string {
    static const std::regex uuid_re(
      R"(([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}))"
    );
    std::smatch m;
    if (std::regex_search(s, m, uuid_re)) {
      std::string u = m[1].str();
      std::transform(u.begin(), u.end(), u.begin(), [](unsigned char c){ return (char)std::tolower(c); });
      return u;
    }
    std::string filtered;
    filtered.reserve(s.size());
    for (unsigned char c : s) {
      if (std::isxdigit(c) || c=='-') filtered.push_back((char)c);
    }
    if (std::regex_search(filtered, m, uuid_re)) {
      std::string u = m[1].str();
      std::transform(u.begin(), u.end(), u.begin(), [](unsigned char c){ return (char)std::tolower(c); });
      return u;
    }
    return {};
  };

  auto need_uuid = [&](const std::string& key, const std::string& seg) -> std::string {
    std::string u = first_uuid(seg);
    if (!u.empty()) return u;
    throw std::runtime_error("Bad or missing UUID for key '" + key + "'");
  };

  {
    std::string op_seg = find_segment("op");
    if (op_seg.empty()) throw std::runtime_error("Missing key 'op'");
    std::string op_clean;
    for (unsigned char c : op_seg) if (std::isalpha(c)) op_clean.push_back((char)std::toupper(c));
    if (op_clean != "POTRF" && op_clean != "TRSM" && op_clean != "SYRK" && op_clean != "GEMM")
      throw std::runtime_error("Unsupported op='" + op_clean + "'");
    r.op = op_clean;
  }

  {
    std::string b_seg = find_segment("B");
    if (b_seg.empty()) throw std::runtime_error("Missing key 'B'");
    long long val = 0;
    size_t i = 0;
    while (i < b_seg.size() && !std::isdigit((unsigned char)b_seg[i])) ++i;
    if (i == b_seg.size()) throw std::runtime_error("Bad integer for 'B'");
    size_t j = i;
    while (j < b_seg.size() && std::isdigit((unsigned char)b_seg[j])) ++j;
    try {
      val = std::stoll(b_seg.substr(i, j - i));
    } catch (...) {
      throw std::runtime_error("Bad integer for 'B'");
    }
    if (val <= 0 || val > INT32_MAX) throw std::runtime_error("Out-of-range 'B'");
    r.B = (int)val;
  }

  auto seg_in   = find_segment("in");
  auto seg_inL  = find_segment("inL");
  auto seg_inA  = find_segment("inA");
  auto seg_inC  = find_segment("inC");
  auto seg_inAi = find_segment("inAi");
  auto seg_inAj = find_segment("inAj");
  auto seg_out  = find_segment("out");

  if (!seg_in.empty())   r.in   = need_uuid("in",   seg_in);
  if (!seg_inL.empty())  r.inL  = need_uuid("inL",  seg_inL);
  if (!seg_inA.empty())  r.inA  = need_uuid("inA",  seg_inA);
  if (!seg_inC.empty())  r.inC  = need_uuid("inC",  seg_inC);
  if (!seg_inAi.empty()) r.inAi = need_uuid("inAi", seg_inAi);
  if (!seg_inAj.empty()) r.inAj = need_uuid("inAj", seg_inAj);
  if (!seg_out.empty())  r.out  = need_uuid("out",  seg_out);

  if (r.op == "POTRF") {
    if (r.in.empty() || r.out.empty()) throw std::runtime_error("POTRF needs 'in' and 'out'");
  } else if (r.op == "TRSM") {
    if (r.inL.empty() || r.inA.empty() || r.out.empty()) throw std::runtime_error("TRSM needs 'inL','inA','out'");
  } else if (r.op == "SYRK") {
    if (r.inC.empty() || r.inA.empty() || r.out.empty()) throw std::runtime_error("SYRK needs 'inC','inA','out'");
  } else if (r.op == "GEMM") {
    if (r.inC.empty() || r.inAi.empty() || r.inAj.empty() || r.out.empty())
      throw std::runtime_error("GEMM needs 'inC','inAi','inAj','out'");
  }

  return r;
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

static std::string clean_id(std::string s) {
  s = std::regex_replace(s, std::regex("[^0-9a-fA-F\\-]"), "");
  return s;
}


static std::string download_blob(armonik::api::worker::TaskHandler& th,
  const std::string& resultId)
{
std::cerr << "[INFO] download_blob: start, id=" << resultId << "\n";
const auto& deps = th.getDataDependencies();
std::cerr << "[INFO] download_blob: deps.size=" << deps.size() << "\n";

  auto it = deps.find(resultId);
  if (it == deps.end()) {
    std::cerr << "[INFO] download_blob: id not found in deps\n";
    throw std::runtime_error("Input '" + resultId + "' not found in data dependencies");
  }
const std::string& path = it->second;

std::cerr << "[INFO] download_blob: found path=" << path << " (exists=" 
            << (std::filesystem::exists(path) ? "yes" : "no") << ")\n";

std::ifstream f(path, std::ios::binary);
if (!f) {
    int e = errno;
    std::cerr << "[INFO] download_blob: open fail, errno=" << e << " (" << std::strerror(e) << ")\n";
    throw std::runtime_error("Cannot open dependency file: " + path +
                             " (id=" + resultId + ", errno=" + std::to_string(e) + " - " + std::strerror(e) + ")");
  }
  std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::cerr << "[INFO] download_blob: read OK, bytes=" << data.size() << "\n";
  return data;
}

static std::string sanitize_utf8_ascii(std::string s) {
  std::string out; out.reserve(s.size());
  for (unsigned char c : s) {
    if (c == '\n' || c == '\r' || (c >= 32 && c < 127)) out.push_back((char)c);
    else out.push_back('?');
  }
  return out;
}



static void upload_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId, const std::string& data) {
  std::cerr << "[INFO] upload_blob: start, id=" << resultId << ", bytes=" << data.size() << "\n";
  th.send_result(resultId, data).get();
  std::cerr << "[INFO] upload_blob: done\n";

}


static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  std::cerr << "[INFO] CHAMELEON_Desc_Create: start, B=" << B << "\n";
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
  std::cerr << "[INFO] CHAMELEON_Desc_Create: done\n";
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


      std::cerr << "[INFO] Execute: begin\n";

      // Log pour voir le mapping
      const auto& deps = taskHandler.getDataDependencies();
      std::cerr << "[INFO] Execute: deps.size=" << deps.size() << "\n";
      for (auto& kvp : deps) {
        std::cerr << "  [INFO] depId=" << kvp.first << " path=" << kvp.second << "\n";
      }

      std::cerr << "[INFO] Execute: getPayload()\n";
      const std::string payload = taskHandler.getPayload();
      std::cerr << "[INFO] Execute: payload bytes=" << payload.size() << "\n";

      // --- Parsing robuste (UUIDs déjà canoniques, 36 chars) ---
      std::cerr << "[INFO] Execute: parse_kv_payload()\n";
      const auto P = parse_kv_payload(payload);  // suppose: struct { std::string op; int B; std::string in,inL,inA,inC,inAi,inAj,out; }
      std::cerr << "[INFO] Execute: op=" << P.op << " B=" << P.B
                << " in="  << P.in  << " inL=" << P.inL  << " inA=" << P.inA
                << " inC=" << P.inC << " inAi="<< P.inAi << " inAj="<< P.inAj
                << " out=" << P.out << "\n";


      // Sécurité : vérifier la présence des IDs requis par op
      //if (r.op == "POTRF") {
      //  if (r.in.empty() || r.out.empty())
      //    throw std::runtime_error("POTRF requires 'in' and 'out' ids");
      //} else if (r.op == "TRSM") {
      //  if (r.inL.empty() || r.inA.empty() || r.out.empty())
      //    throw std::runtime_error("TRSM requires 'inL', 'inA' and 'out' ids");
      //} else if (r.op == "SYRK") {
      //  if (r.inC.empty() || r.inA.empty() || r.out.empty())
      //    throw std::runtime_error("SYRK requires 'inC', 'inA' and 'out' ids");
      //} else if (r.op == "GEMM") {
      //  if (r.inC.empty() || r.inAi.empty() || r.inAj.empty() || r.out.empty())
      //    throw std::runtime_error("GEMM requires 'inC', 'inAi', 'inAj' and 'out' ids");
      //} else {
      //  return armonik::api::worker::ProcessStatus("Unknown op=" + r.op);
      //}

      const std::string& op = P.op;
      const int B = P.B;

      double secs = 0.0, flops = 0.0, gflops = 0.0;
      struct timespec t0{}, t1{};

      // ============================
      // POTRF
      // ============================
      if (P.op == "POTRF") {
        std::cerr << "[INFO] POTRF: download A, id=" << P.in << "\n";
        const auto blobA = download_blob(taskHandler, P.in);
        std::cerr << "[INFO] POTRF: deserialize A\n";
        std::vector<double> A = deserialize_block(blobA, B);

        CHAM_desc_t *dA = nullptr;
        std::cerr << "[INFO] POTRF: create desc dA\n";
        create_desc_1block(&dA, A.data(), B);

        std::cerr << "[INFO] POTRF: compute\n";
        int info = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) {
          CHAMELEON_Desc_Destroy(&dA);
          throw std::runtime_error("dpotrf info=" + std::to_string(info));
        }

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        std::cerr << "[INFO] POTRF: upload out=" << P.out << "\n";
        upload_blob(taskHandler, P.out, serialize_block(A));

        CHAMELEON_Desc_Destroy(&dA);
      }

      // ============================
      // TRSM
      // ============================
      else if (P.op == "TRSM") {
        std::cerr << "[INFO] TRSM: download L, id=" << P.inL << "\n";
        const auto blobL = download_blob(taskHandler, P.inL);
        std::vector<double> L = deserialize_block(blobL, B);

        std::cerr << "[INFO] TRSM: download A, id=" << P.inA << "\n";
        const auto blobA = download_blob(taskHandler, P.inA);
        std::vector<double> A = deserialize_block(blobA, B);

        CHAM_desc_t *dL = nullptr, *dA = nullptr;
        std::cerr << "[INFO] TRSM: create desc dL,dA\n";
        create_desc_1block(&dL, L.data(), B);
        create_desc_1block(&dA, A.data(), B);

        std::cerr << "[INFO] TRSM: compute\n";
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 0.5 * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        std::cerr << "[INFO] TRSM: upload out=" << P.out << "\n";
        upload_blob(taskHandler, P.out, serialize_block(A));

        CHAMELEON_Desc_Destroy(&dL);
        CHAMELEON_Desc_Destroy(&dA);
      }

      // ============================
      // SYRK
      // ============================
      else if (P.op == "SYRK") {
        std::cerr << "[INFO] SYRK: download C, id=" << P.inC << "\n";
        const std::string blobC = download_blob(taskHandler, P.inC);
        std::vector<double> C = deserialize_block(blobC, B);

        std::cerr << "[INFO] SYRK: download A, id=" << P.inA << "\n";
        const std::string blobA = download_blob(taskHandler, P.inA);
        std::vector<double> A = deserialize_block(blobA, B);

        CHAM_desc_t *dC = nullptr, *dA = nullptr;
        std::cerr << "[INFO] SYRK: create desc dC,dA\n";
        create_desc_1block(&dC, C.data(), B);
        create_desc_1block(&dA, A.data(), B);

        std::cerr << "[INFO] SYRK: compute\n";
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        std::cerr << "[INFO] SYRK: upload out=" << P.out << "\n";
        upload_blob(taskHandler, P.out, serialize_block(C));

        CHAMELEON_Desc_Destroy(&dC);
        CHAMELEON_Desc_Destroy(&dA);
      }

      // ============================
      // GEMM
      // ============================
      else if (P.op == "GEMM") {
        std::cerr << "[INFO] GEMM: download C, id=" << P.inC << "\n";
        const std::string blobC = download_blob(taskHandler, P.inC);
        std::vector<double> C = deserialize_block(blobC, B);

        std::cerr << "[INFO] GEMM: download Ai, id=" << P.inAi << "\n";
        const std::string blobAi = download_blob(taskHandler, P.inAi);
        std::vector<double> Ai = deserialize_block(blobAi, B);

        std::cerr << "[INFO] GEMM: download Aj, id=" << P.inAj << "\n";
        const std::string blobAj = download_blob(taskHandler, P.inAj);
        std::vector<double> Aj = deserialize_block(blobAj, B);

        CHAM_desc_t *dC = nullptr, *dAi = nullptr, *dAj = nullptr;
        std::cerr << "[INFO] GEMM: create desc dC,dAi,dAj\n";
        create_desc_1block(&dC,  C.data(),  B);
        create_desc_1block(&dAi, Ai.data(), B);
        create_desc_1block(&dAj, Aj.data(), B);

        std::cerr << "[INFO] GEMM: compute\n";
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 2.0 * (double)B * (double)B * (double)B;
        gflops = flops / (secs * 1e9);

        std::cerr << "[INFO] GEMM: upload out=" << P.out << "\n";
        upload_blob(taskHandler, P.out, serialize_block(C));

        CHAMELEON_Desc_Destroy(&dC);
        CHAMELEON_Desc_Destroy(&dAi);
        CHAMELEON_Desc_Destroy(&dAj);
      }

      else {
        return armonik::api::worker::ProcessStatus("Unknown op=" + P.op);
      }

      std::cerr << "[INFO] Execute: end OK\n";
      std::cerr << "[PERF] " << P.op << " B=" << B
                << " time=" << secs << " sec"
                << " flops=" << flops << " gflops=" << gflops << "\n";

      return armonik::api::worker::ProcessStatus::Ok;
    } catch (const std::exception &e) {
      const std::string msg = sanitize_utf8_ascii(e.what());
      std::cerr << "[DagWorker] ERROR: " << msg << "\n";
      return armonik::api::worker::ProcessStatus(msg);
    }
  }
};
      //std::cerr << "[INFO] Execute: getPayload()\n";
      //const std::string payload = taskHandler.getPayload();
      //std::cerr << "[INFO] Execute: payload bytes=" << payload.size() << "\n";
      
      //std::cerr << "[INFO] Execute: parse_kv_payload()\n";
      //auto kv = parse_kv_payload(payload);
      //std::cerr << "[INFO] Execute: kv.size=" << kv.size() << "\n";

      //const std::string op = to_str(kv, "op");
      //const int B = to_int(kv, "B");
      //std::cerr << "[INFO] Execute: op=" << op << " B=" << B << "\n";

      // Récupération des IDs d'entrée/sortie depuis le payload
      //std::string in, inL, inA, inC, inAi, inAj, out;
      //if (op == "POTRF") { in = to_str(kv, "in"); out = to_str(kv, "out"); }
      //else if (op == "TRSM") { inL = to_str(kv, "inL"); inA = to_str(kv, "inA"); out = to_str(kv, "out"); }
      //else if (op == "SYRK") { inC = to_str(kv, "inC"); inA = to_str(kv, "inA"); out = to_str(kv, "out"); }
      //else if (op == "GEMM") { inC = to_str(kv, "inC"); inAi= to_str(kv, "inAi"); inAj= to_str(kv, "inAj"); out= to_str(kv, "out"); }
      //else return armonik::api::worker::ProcessStatus("Unknown op=" + op);

      //std::cerr << "[INFO] Execute: raw IDs: in="<<in<<" inL="<<inL<<" inA="<<inA
      //          <<" inC="<<inC<<" inAi="<<inAi<<" inAj="<<inAj<<" out="<<out<<"\n";

      // Nettoyage des IDs
      //std::cerr << "[INFO] Execute: clean_id()\n";
      //in   = clean_id(in);
      //inL  = clean_id(inL);
      //inA  = clean_id(inA);
      //inC  = clean_id(inC);
      //inAi = clean_id(inAi);
      //inAj = clean_id(inAj);
      //out  = clean_id(out);

      //std::cerr << "[INFO] Execute: IDs cleaned: in="<<in<<" inL="<<inL<<" inA="<<inA
      //          <<" inC="<<inC<<" inAi="<<inAi<<" inAj="<<inAj<<" out="<<out<<"\n";

      //int info = 0;
      //std::vector<double> A(B*B), L(B*B), C(B*B), Ai(B*B), Aj(B*B);
      //CHAM_desc_t *dA=nullptr, *dL=nullptr, *dC=nullptr, *dAi=nullptr, *dAj=nullptr;


      //if (op == "POTRF") {
      //  std::cerr << "[INFO] POTRF: download A, id=" << in << "\n";
      //  auto blobA = download_blob(taskHandler, in);
      //  std::cerr << "[INFO] POTRF: deserialize A\n";
      //  A = deserialize_block(blobA, B);
      //  std::cerr << "[INFO] POTRF: create desc dA\n";
      //  create_desc_1block(&dA, A.data(), B);

      //} else if (op == "TRSM") {
      //  std::cerr << "[INFO] TRSM: download L, id=" << inL << "\n";
      //  auto blobL = download_blob(taskHandler, inL);
      //  std::cerr << "[INFO] TRSM: deserialize L\n";
      //  L = deserialize_block(blobL, B);
      //  std::cerr << "[INFO] TRSM: download A, id=" << inA << "\n";
      //  auto blobA = download_blob(taskHandler, inA);
      //  std::cerr << "[INFO] TRSM: deserialize A\n";
      //  A = deserialize_block(blobA, B);
      //  std::cerr << "[INFO] TRSM: create desc dL,dA\n";
      //  create_desc_1block(&dL, L.data(), B);
      //  create_desc_1block(&dA, A.data(), B);

      //} else if (op == "SYRK") {
      //  std::cerr << "[INFO] SYRK: download C, id=" << inC << "\n";
      //  auto blobC = download_blob(taskHandler, inC);
      //  std::cerr << "[INFO] SYRK: deserialize C\n";
      //  C = deserialize_block(blobC, B);
      //  std::cerr << "[INFO] SYRK: download A, id=" << inA << "\n";
      //  auto blobA = download_blob(taskHandler, inA);
      //  std::cerr << "[INFO] SYRK: deserialize A\n";
      //  A = deserialize_block(blobA, B);
      //  std::cerr << "[INFO] SYRK: create desc dC,dA\n";
      //  create_desc_1block(&dC, C.data(), B);
      //  create_desc_1block(&dA, A.data(), B);

      //} else { // GEMM
      //  std::cerr << "[INFO] GEMM: download C, id=" << inC << "\n";
      //  auto blobC = download_blob(taskHandler, inC);
      //  std::cerr << "[INFO] GEMM: deserialize C\n";
      //  C = deserialize_block(blobC, B);
      //  std::cerr << "[INFO] GEMM: download Ai, id=" << inAi << "\n";
      //  auto blobAi = download_blob(taskHandler, inAi);
      //  std::cerr << "[INFO] GEMM: deserialize Ai\n";
      //  Ai = deserialize_block(blobAi, B);
      //  std::cerr << "[INFO] GEMM: download Aj, id=" << inAj << "\n";
      //  auto blobAj = download_blob(taskHandler, inAj);
      //  std::cerr << "[INFO] GEMM: deserialize Aj\n";
      //  Aj = deserialize_block(blobAj, B);
      //  std::cerr << "[INFO] GEMM: create desc dC,dAi,dAj\n";
      //  create_desc_1block(&dC, C.data(), B);
      //  create_desc_1block(&dAi, Ai.data(), B);
      //  create_desc_1block(&dAj, Aj.data(), B);
      //}

      //struct timespec t0{}, t1{};
      //auto Bd = static_cast<double>(B);
      //double secs = 0.0, flops = 0.0, gflops = 0.0;


      //if (op == "POTRF") {
      //  clock_gettime(CLOCK_MONOTONIC, &t0);
      //  info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
      //  clock_gettime(CLOCK_MONOTONIC, &t1);
      //  if (info != 0) throw std::runtime_error("dpotrf info=" + std::to_string(info));
      //  secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
      //  flops = (1.0/3.0) * Bd*Bd*Bd; (void)gflops; gflops = flops / (secs * 1e9);
      //  upload_blob(taskHandler, out, serialize_block(A));
      //}
      //else if (op == "TRSM") {
      //  clock_gettime(CLOCK_MONOTONIC, &t0);
      //  CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
      //  clock_gettime(CLOCK_MONOTONIC, &t1);
      //  secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
      //  flops = 0.5 * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
      //  upload_blob(taskHandler, out, serialize_block(A));
      //}
      //else if (op == "SYRK") {
      //  clock_gettime(CLOCK_MONOTONIC, &t0);
      //  CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
      //  clock_gettime(CLOCK_MONOTONIC, &t1);
      //  secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
      //  flops = Bd*Bd*Bd; gflops = flops / (secs * 1e9);
      //  upload_blob(taskHandler, out, serialize_block(C));
      //}

      //else { // GEMM
      //  clock_gettime(CLOCK_MONOTONIC, &t0);
      //  CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
      //  clock_gettime(CLOCK_MONOTONIC, &t1);
      //  secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
      //  flops = 2.0 * Bd*Bd*Bd; gflops = flops / (secs * 1e9);
      //  upload_blob(taskHandler, out, serialize_block(C));
      //}


      //std::cerr << "[INFO] Destroy descriptors start\n";
      //if (dA)  { CHAMELEON_Desc_Destroy(&dA);  std::cerr << "[INFO] dA destroyed\n"; }
      //if (dL)  { CHAMELEON_Desc_Destroy(&dL);  std::cerr << "[INFO] dL destroyed\n"; }
      //if (dC)  { CHAMELEON_Desc_Destroy(&dC);  std::cerr << "[INFO] dC destroyed\n"; }
      //if (dAi) { CHAMELEON_Desc_Destroy(&dAi); std::cerr << "[INFO] dAi destroyed\n"; }
      //if (dAj) { CHAMELEON_Desc_Destroy(&dAj); std::cerr << "[INFO] dAj destroyed\n"; }
      //std::cerr << "[INFO] Destroy descriptors done\n";
      
      // CHAMELEON_Finalize();

      //std::cerr << "[INFO] Execute: end OK\n";
      //std::cerr << "[PERF] " << op << " B=" << B 
      //          << " time=" << secs << " sec"
      //          << " flops=" << flops << " gflops=" << gflops << "\n";
    //  return armonik::api::worker::ProcessStatus::Ok;
    //}
    //catch (const std::exception& e) {

      // Ajout de la sanitise
    //  std::string msg = sanitize_utf8_ascii(e.what());

    //  std::cerr << "[DagWorker] ERROR: " << msg << "\n";
    //  return armonik::api::worker::ProcessStatus(msg);
    //}
  //}
//};

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
    std::cerr << "[INFO] main: starting WorkerServer::run()\n";
    armonik::api::worker::WorkerServer::create<DagCholeskyWorker>(config)->run();
    std::cerr << "[INFO] main: WorkerServer::run() returned\n";
  } catch (const std::exception &e) {
    std::cerr << "Error in worker: " << e.what() << std::endl;
  }


  std::cout << "Stopping Server..." << std::endl;
  return 0;
}



// ------------------------- Utils -------------------------------------------------
static int env_int(const char* key, int defval) {
  if (const char* s = std::getenv(key)) { try { return std::max(0, std::stoi(s)); } catch (...) {} }
  return defval;
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
static std::string sanitize_utf8_ascii(std::string s) {
  std::string out; out.reserve(s.size());
  for (unsigned char c : s) out.push_back((c=='\n'||c=='\r'||(c>=32&&c<127))?(char)c:'?');
  return out;
}

static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
}
struct DescRAII {
  CHAM_desc_t* p=nullptr; ~DescRAII(){ if(p) CHAMELEON_Desc_Destroy(&p); }
};

// ------------------------- Payload parsing (simple & robuste) --------------------
struct ParsedPayload {
  std::string op; // POTRF/TRSM/SYRK/GEMM
  int B = 0;
  std::string in, inL, inA, inC, inAi, inAj, out; // UUIDs (36 chars)
};

static bool is_uuid36(std::string s) {
  static const std::regex re(
    R"(^[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}$)"
  );
  return std::regex_match(s, re);
}

static std::string to_lower(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c);});
  return s;
}
static std::string to_upper(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::toupper(c);});
  return s;
}

static ParsedPayload parse_kv_payload(const std::string& s) {
  // Tokenise "key=value" séparés par espaces (pas d'échappement, volontairement simple).
  std::map<std::string,std::string> kv;
  for (size_t i=0; i<s.size();) {
    while (i<s.size() && std::isspace((unsigned char)s[i])) ++i;
    if (i>=s.size()) break;
    size_t j=i; while (j<s.size() && !std::isspace((unsigned char)s[j])) ++j;
    std::string tok = s.substr(i, j-i);
    size_t eq = tok.find('=');
    if (eq!=std::string::npos && eq>0 && eq+1<tok.size()) {
      std::string k = tok.substr(0, eq);
      std::string v = tok.substr(eq+1);
      kv[k]=v;
    }
    i=j;
  }

  auto need = [&](const char* key)->std::string {
    auto it = kv.find(key);
    if (it==kv.end() || it->second.empty())
      throw std::runtime_error(std::string("Missing key '")+key+"'");
    return it->second;
  };
  auto opt  = [&](const char* key)->std::string {
    auto it = kv.find(key); return it==kv.end()?std::string():it->second;
  };

  ParsedPayload p;
  p.op = to_upper(need("op"));
  if (p.op!="POTRF" && p.op!="TRSM" && p.op!="SYRK" && p.op!="GEMM")
    throw std::runtime_error("Unsupported op='"+p.op+"'");

  try { p.B = std::stoi(need("B")); } catch(...) { throw std::runtime_error("Bad integer for 'B'"); }
  if (p.B<=0) throw std::runtime_error("B must be > 0");

  auto take_uuid = [&](const char* key, bool required)->std::string {
    std::string v = opt(key);
    if (v.empty()) {
      if (required) throw std::runtime_error(std::string("Missing key '")+key+"'");
      return {};
    }
    v = to_lower(v);
    if (!is_uuid36(v)) throw std::runtime_error(std::string("Invalid UUID for '")+key+"'");
    return v;
  };

  if (p.op=="POTRF") {
    p.in  = take_uuid("in",  true);
    p.out = take_uuid("out", true);
  } else if (p.op=="TRSM") {
    p.inL = take_uuid("inL", true);
    p.inA = take_uuid("inA", true);
    p.out = take_uuid("out", true);
  } else if (p.op=="SYRK") {
    p.inC = take_uuid("inC", true);
    p.inA = take_uuid("inA", true);
    p.out = take_uuid("out", true);
  } else { // GEMM
    p.inC = take_uuid("inC",  true);
    p.inAi= take_uuid("inAi", true);
    p.inAj= take_uuid("inAj", true);
    p.out = take_uuid("out",  true);
  }

  return p;
}

// ------------------------- I/O via TaskHandler -----------------------------------
static std::string download_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId) {
  // Map (resultId -> chemin local)
  const auto& deps = th.getDataDependencies();
  auto it = deps.find(resultId);
  if (it == deps.end())
    throw std::runtime_error("Input '"+resultId+"' not found in data dependencies");

  const std::string& path = it->second;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    int e = errno;
    throw std::runtime_error("Cannot open dependency file: " + path +
                              " (id=" + resultId + ", errno=" + std::to_string(e) + " - " + std::string(std::strerror(e)) + ")");
  }
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void upload_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId, const std::string& data) {
  th.send_result(resultId, data).get(); // envoi synchrone
}

// ------------------------- Worker ------------------------------------------------
class DagCholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent)
  : ArmoniKWorker(std::move(agent)) {}

  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    try {
      const std::string payload = taskHandler.getPayload();
      const ParsedPayload P = parse_kv_payload(payload);

      // Pour debug : montrer les deps (id -> chemin)
      // for (auto& kv : taskHandler.getDataDependencies())
      //   std::cerr << "[dep] " << kv.first << " -> " << kv.second << "\n";

      const int B = P.B;
      struct timespec t0{}, t1{};
      double secs=0.0, flops=0.0;

      if (P.op=="POTRF") {
        auto Ablob = download_blob(taskHandler, P.in);
        std::vector<double> A = deserialize_block(Ablob, B);

        DescRAII dA; create_desc_1block(&dA.p, A.data(), B);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int info = CHAMELEON_dpotrf_Tile(ChamLower, dA.p);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info!=0) throw std::runtime_error("dpotrf info="+std::to_string(info));

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        flops = (1.0/3.0)*B*B*B;
        upload_blob(taskHandler, P.out, serialize_block(A));
      }
      else if (P.op=="TRSM") {
        auto Lblob = download_blob(taskHandler, P.inL);
        auto Ablob = download_blob(taskHandler, P.inA);
        std::vector<double> L = deserialize_block(Lblob, B);
        std::vector<double> A = deserialize_block(Ablob, B);

        DescRAII dL, dA; create_desc_1block(&dL.p, L.data(), B); create_desc_1block(&dA.p, A.data(), B);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL.p, dA.p);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        flops = 0.5*B*B*B;
        upload_blob(taskHandler, P.out, serialize_block(A));
      }
      else if (P.op=="SYRK") {
        auto Cblob = download_blob(taskHandler, P.inC);
        auto Ablob = download_blob(taskHandler, P.inA);
        std::vector<double> C = deserialize_block(Cblob, B);
        std::vector<double> A = deserialize_block(Ablob, B);

        DescRAII dC, dA; create_desc_1block(&dC.p, C.data(), B); create_desc_1block(&dA.p, A.data(), B);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA.p, 1.0, dC.p);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        flops = 1.0*B*B*B;
        upload_blob(taskHandler, P.out, serialize_block(C));
      }
      else { // GEMM
        auto Cblob  = download_blob(taskHandler, P.inC);
        auto Aiblob = download_blob(taskHandler, P.inAi);
        auto Ajblob = download_blob(taskHandler, P.inAj);
        std::vector<double> C  = deserialize_block(Cblob,  B);
        std::vector<double> Ai = deserialize_block(Aiblob, B);
        std::vector<double> Aj = deserialize_block(Ajblob, B);

        DescRAII dC,dAi,dAj;
        create_desc_1block(&dC.p,  C.data(),  B);
        create_desc_1block(&dAi.p, Ai.data(), B);
        create_desc_1block(&dAj.p, Aj.data(), B);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi.p, dAj.p, 1.0, dC.p);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        flops = 2.0*B*B*B;
        upload_blob(taskHandler, P.out, serialize_block(C));
      }

      std::cerr << "[PERF] " << P.op << " B=" << B
                << " time=" << secs << " s"
                << " gflops=" << (flops/(secs*1e9)) << "\n";

      return armonik::api::worker::ProcessStatus::Ok;
    } catch (const std::exception& e) {
      return armonik::api::worker::ProcessStatus(sanitize_utf8_ascii(e.what()));
    }
  }
};

// ------------------------- Main --------------------------------------------------
int main() {
  std::cout << "DagCholeskyWorker started. gRPC=" << grpc::Version() << "\n";

  // Init CHAMELEON (process-wide)
  int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency()));
  int ngpu = env_int("CHM_NGPU", 0);
  CHAMELEON_Init(ncpu, ngpu);
  std::atexit([]{ CHAMELEON_Finalize(); });

  armonik::api::common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  config.set("ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock");
  config.set("ComputePlane__AgentChannel__Address",  "/cache/armonik_agent.sock");

  try {
    armonik::api::worker::WorkerServer::create<DagCholeskyWorker>(config)->run();
  } catch (const std::exception &e) {
    std::cerr << "Worker fatal error: " << e.what() << "\n";
  }
  return 0;
}
