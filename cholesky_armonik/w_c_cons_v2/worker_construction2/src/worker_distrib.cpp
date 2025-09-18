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

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "absl/strings/string_view.h"

#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"

#include <sstream>

#include <chameleon.h>   

// ============================================================================
// Serialize and Deserialize function
// ============================================================================

struct Parsed {std::string op; int B = 0; std::string in, inL, inA, inC, inAi, inAj; };
static Parsed handle_json(const std::string& payload) {
    rapidjson::Document d; d.Parse(payload.c_str());

    Parsed p;
    p.op = std::string{ d["op"].GetString()};
    p.B  = d["B"].GetInt();

    if (p.op == "POTRF") {
        p.in = std::string{ d["in"].GetString()};
    } else if (p.op == "TRSM") {
        p.inL = std::string{ d["inL"].GetString()};
        p.inA = std::string{ d["inA"].GetString()};
    } else if (p.op == "SYRK") {
        p.inC = std::string{ d["inC"].GetString()};
        p.inA = std::string{ d["inA"].GetString()};
    } else if (p.op == "GEMM") {
        p.inC  = std::string{ d["inC"].GetString()};
        p.inAi = std::string{ d["inAi"].GetString()};
        p.inAj = std::string{ d["inAj"].GetString()};
    }
    return p;
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
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent): ArmoniKWorker(std::move(agent)) {}
  
  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    
    // std::cout << " [Worker] 1st line in execute loop " << std::endl  ;

    try {
      // std::cout << " [Worker] 1st line in try loop " << std::endl ;
      const std::string payload = taskHandler.getPayload();
      std::cout << " [Worker][.getPayload] : payload_json " << payload << std::endl ;
      Parsed p = handle_json(payload);
      /*
      std::cout << " [Worker][payload parse] : op="  << p.op << std::endl;
      std::cout << " [Worker][payload parse] : B="   << p.B << std::endl ;
      std::cout << " [Worker][payload parse] : in="  << p.in  << std::endl;
      std::cout << " [Worker][payload parse] : inL=" << p.inL << std::endl ;
      std::cout << " [Worker][payload parse] : inA=" << p.inA << std::endl ;
      std::cout << " [Worker][payload parse] : inC=" << p.inC << std::endl;
      std::cout << " [Worker][payload parse] : inAi="<< p.inAi << std::endl ;
      std::cout << " [Worker][payload parse] : inAj="<< p.inAj<< std::endl ;
      */ 

      // --- Helpers debug locaux (col-major supposé, ld = B) ---
      auto expect_bytes = [&](int B){ return (size_t)B * (size_t)B * sizeof(double); };

      auto frob = [&](const std::vector<double>& X){
        long double s = 0.0L; for (double v : X) s += (long double)v * (long double)v;
        return (double)std::sqrt(s);
      };

      auto naninf = [&](const std::vector<double>& X){
        size_t n = 0; for (double v : X) if (!std::isfinite(v)) ++n; return n;
      };

      auto diag_min = [&](const std::vector<double>& X, int B){
        double mn = std::numeric_limits<double>::infinity();
        for (int i=0;i<B;++i) mn = std::min(mn, X[(size_t)i + (size_t)i*B]); // col-major
        return mn;
      };

      auto upper_maxabs = [&](const std::vector<double>& X, int B){
        double m = 0.0;
        for (int j=0;j<B;++j) for (int i=0;i<j;++i)
          m = std::max(m, std::abs(X[(size_t)i + (size_t)j*B])); // au-dessus de la diag
        return m;
      };

      auto dump4 = [&](const char* tag, const std::vector<double>& X){
        std::cout << tag << " first4=";
        for (int i=0; i<std::min<int>(4,(int)X.size()); ++i) std::cout << X[i] << " ";
        std::cout << "\n";
      };



      double secs = 0.0, flops = 0.0, gflops = 0.0;
      struct timespec t0{}, t1{};

        // Ce qui est attendu
        //           expected
        //      -    * expected = taskHandler.getExpectedResults() renvoie vers un std::vector<std::string>, Map { resultId → chemin local }
        //           * donc faire const std::string& out_id = expected.front(); pour avoir le bon type
        //           * où out-id est l'ID de sortie attendu
        //
        //           deps
        //      -    * const auto& deps = taskHandler.getDataDependencies(); renvoie vers un std::map<std::string, std::string>
        //           * tel que par ex : deps = { "b7528962-6662-40a6-a9f7-3a3d88a5eab7" : "< octets binaires>"}
        //
        //           it
        //      -    * auto it = deps.find(p.in); renvoie un itérateur dont le but est de retrouver la paire { key = p.in, value = … } si elle existe.
        //           * tel que it = deps.find(p.in) pointe sur la paire :
        //                           - key   = "b7528962-6662-40a6-a9f7-3a3d88a5eab7"
        //                           - value = "< octets binaires>"
        //
        //           in_blob
        //      -    * une référence vers la valeur de la paire pointée par it (donc les octets du bloc d’entrée)
        //           * ça contient les octets serialize_1_block(A) envoyés/stockés côté client pour ce bloc.


      // ============================
      // POTRF
      // ============================
      if (p.op == "POTRF") {        
        const auto&         expected = taskHandler.getExpectedResults();
        std::cout << " [Worker][POTF][expected = taskHandler.getExpectedResults()] : expected size ="  << expected.size() << std::endl ;

        const std::string&    out_id = expected.front();
        // std::cout << " [Worker][POTF][out_id] : out_id ="  << out_id << std::endl;

        const auto&             deps = taskHandler.getDataDependencies();
        /*
        for (const auto& kv : deps) {
          std::cout << " [WORKER][POTF][DEP] id =" << kv.first << std::endl;
          std::cout << " [WORKER][POTF][DEP] bytes=" << kv.second.size() << std::endl;
}       */

        auto                      it = deps.find(p.in);
        if (it == deps.end())
          return armonik::api::worker::ProcessStatus("[Worker][POTF] Missing dependency: " + p.in);     // P.in est l'ID du bloc d'entrée
        
        const std::string&   in_blob = it->second;
        // std::cout << "[WORKER][POTF][in_blob = it] : in_blob =" << in_blob << std::endl;
        // std::cout << "[WORKER][POTF][in_blob = it] : in_blob size =" << in_blob.size() << std::endl;
        // std::cout << "[WORKER][POTF][in_blob = it] : in_blob byte expected =" << (p.B*p.B*sizeof(double)) << std::endl;

        // dump rapide
        /*
        size_t nshow = std::min<size_t>(4, in_blob.size()/sizeof(double));
        const double* d = reinterpret_cast<const double*>(in_blob.data());
        std::cout << "[WORKER][POTRF] first_doubles=";
        for (size_t i=0;i<nshow;++i) std::cout << d[i] << " ";
          std::cout << std::endl;
        */

        std::cout << "[WORKER][POTF] : start dezerialisation" << std::endl ;
        const double* ptr = reinterpret_cast<const double*>(in_blob.data());
        std::vector<double> A(ptr, ptr + in_blob.size()/sizeof(double));
        // std::vector<double> A = deserialize_1_block(in_blob);
        std::cout << "[WORKER][POTF] : end dezerialisation" << std::endl;

        // Check size
        if ((int)A.size() != p.B * p.B) {
          return armonik::api::worker::ProcessStatus("[Worker][POTF] Bad block size: expected " + std::to_string(p.B*p.B) +" doubles, got " + std::to_string(A.size()));
        }

        CHAM_desc_t *dA = nullptr;

        create_desc_1block(&dA, A.data(), p.B);
        
        // --- DEBUG: afficher la diagonale ---
        auto diag_at = [&](int i){ return A[i + i*p.B]; }; // col-major: a(i,i) = A[i + i*ld], ld=B
        double mindiag = std::numeric_limits<double>::infinity();
        for (int i = 0; i < p.B; ++i) {
          mindiag = std::min(mindiag, diag_at(i));
        }
        std::cout << "[WORKER][POTRF] diag min=" << mindiag << std::endl;
        // ------------------------------------

        int info = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        // std::cout << "[WORKER][POTF] : chameleon potrf start" << std::endl;
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        //std::cout << "[WORKER][POTF] : chameleon potrf end  " << std::endl;
        std::cout << "[WORKER][POTF] : chameleon potrf info " << std::to_string(info) << std::endl;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) {CHAMELEON_Desc_Destroy(&dA);
          throw std::runtime_error("[Worker][POTF] dpotrf info=" + std::to_string(info));}

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * (double)p.B * (double)p.B * (double)p.B;
        gflops = flops / (secs * 1e9);

        std::cout << "[WORKER][POTF] : start double -> string_view" << std::endl;
        absl::string_view out_blockView(reinterpret_cast<const char*>(A.data()),A.size() * sizeof(double));
        std::cout << "[WORKER][POTF] out_blockView.size =" << out_blockView.size() << std::endl;
        // std::cout << "[WORKER][POTF] : end double -> string_view" << std::endl;

        // const std::string out_blob = serialize_1_block(A);
        CHAMELEON_Desc_Destroy(&dA);

        try {
              // taskHandler.send_result(out_id, out_blob).get();
              std::cout << "[WORKER][POTF] : start send_resukt" << std::endl;
              taskHandler.send_result(out_id, out_blockView).get();
              std::cout << "[WORKER][POTF] : end send_result" << std::endl;
            } catch (const std::exception& e) 
            {
                return armonik::api::worker::ProcessStatus(std::string("[Worker][POTF] send_result failed: ") + e.what());
            }
            return armonik::api::worker::ProcessStatus::Ok;
        }

      // ============================
      // TRSM
      // ============================
      else if (p.op == "TRSM") {
        const auto& expected = taskHandler.getExpectedResults();
        std::cout << " [Worker][TRSM][expected = taskHandler.getExpectedResults()] : expected size ="  << expected.size() << std::endl ;
        const std::string& out_id = expected.front();
        const auto& deps = taskHandler.getDataDependencies();

        auto itL = deps.find(p.inL);
        if (itL == deps.end())
          return armonik::api::worker::ProcessStatus("[Worker][TRSM] Missing dependency: " + p.inL);
        const std::string& Lblob = itL->second;

        auto itA = deps.find(p.inA);
        if (itA == deps.end())
          return armonik::api::worker::ProcessStatus("[Worker][TRSM] Missing dependency: " + p.inA);
        const std::string& Ablob = itA->second;

        std::cout << "[WORKER][TRSM] : start dezerialisation L" << std::endl ;
        const double* ptrL = reinterpret_cast<const double*>(Lblob.data());
        std::vector<double> L(ptrL, ptrL + Lblob.size()/sizeof(double));
        std::cout << "[WORKER][TRSM] : end dezerialisation L" << std::endl ;

        std::cout << "[WORKER][TRSM] : start dezerialisation A" << std::endl ;
        const double* ptrA = reinterpret_cast<const double*>(Ablob.data());
        std::vector<double> A(ptrA, ptrA + Ablob.size()/sizeof(double));
        std::cout << "[WORKER][TRSM] : end dezerialisation A" << std::endl ;


        dump4("[Worker][TRSM] L pre", L);
        dump4("[Worker][TRSM] A pre", A);
        double L_diag_min = diag_min(L, p.B);
        double L_upper    = upper_maxabs(L, p.B);
        double fnA0 = frob(A), fnL = frob(L);
        size_t nNaNL = naninf(L), nNaNA0 = naninf(A);
        std::cout <<"[WORKER][TRSM]" << " ||L||F=" << fnL
                  << " / diag(L)_min=" << L_diag_min
                  << " / upper(L)_maxabs=" << L_upper
                  << " / NaN/Inf L=" << nNaNL
                  << " / ||A||F0=" << fnA0
                  << " / NaN/Inf A0=" << nNaNA0
                  << std::endl;

        // std::vector<double> L = deserialize_1_block(Lblob);
        // std::vector<double> A = deserialize_1_block(Ablob);

        CHAM_desc_t *dL = nullptr, *dA = nullptr;
        create_desc_1block(&dL, L.data(), p.B);
        create_desc_1block(&dA, A.data(), p.B);

        int info_trsm =0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info_trsm = CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        std::cout << "[WORKER][TRSM] : chameleon trsm info " << std::to_string(info_trsm) << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (info_trsm != 0) {CHAMELEON_Desc_Destroy(&dA);
                            CHAMELEON_Desc_Destroy(&dL);
          throw std::runtime_error("[Worker][TRSM] dtrsm info=" + std::to_string(info_trsm));}
        
        secs   = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops  = 0.5 * (double)p.B * (double)p.B * (double)p.B;
        gflops = flops / (secs * 1e9);
        
        
        dump4("[TRSM] A post", A);
        double fnA1 = frob(A);
        size_t nNaNA1 = naninf(A);
        std::cout << "[WORKER][TRSM]"<<" ||A||F: " << fnA0 << " -> " << fnA1
                  << " (ratio=" << fnA1/(fnA0+1e-300) << ")"
                  << " NaN/Inf A: " << nNaNA0 << " -> " << nNaNA1
                  << " time=" << secs << "s gflops=" << gflops << std::endl;

        absl::string_view out_blockViewA(reinterpret_cast<const char*>(A.data()),A.size() * sizeof(double));


        // const std::string out_blob = serialize_1_block(A);

        CHAMELEON_Desc_Destroy(&dL);
        CHAMELEON_Desc_Destroy(&dA);

          try {
              // taskHandler.send_result(out_id, out_blob).get();
              taskHandler.send_result(out_id, out_blockViewA).get();
              
            } catch (const std::exception& e) {
              return armonik::api::worker::ProcessStatus(std::string("send_result failed: ") + e.what());
            }
            return armonik::api::worker::ProcessStatus::Ok;
      }

      // ============================
      // SYRK
      // ============================
      else if (p.op == "SYRK") {

        const auto& expected = taskHandler.getExpectedResults();
        std::cout << " [Worker][SYRK][expected = taskHandler.getExpectedResults()] : expected size ="  << expected.size() << std::endl ;
        const std::string& out_id = expected.front();
        const auto& deps = taskHandler.getDataDependencies();

        auto itC = deps.find(p.inC);
        if (itC == deps.end())
          return armonik::api::worker::ProcessStatus(" [Worker][SYRK]Missing dependency: " + p.inC);
        const std::string& Cblob = itC->second;
        
        auto itA = deps.find(p.inA);
          if (itA == deps.end())
            return armonik::api::worker::ProcessStatus("[Worker][SYRK] Missing dependency: " + p.inA);
        const std::string& Ablob = itA->second;

        const double* ptrC = reinterpret_cast<const double*>(Cblob.data());
        std::vector<double> C(ptrC, ptrC + Cblob.size()/sizeof(double)); 

        const double* ptrA = reinterpret_cast<const double*>(Ablob.data());
        std::vector<double> A(ptrA, ptrA + Ablob.size()/sizeof(double));

        
        dump4("[Worker][SYRK] C pre", C);
        dump4("[Worker][SYRK] A pre", A);
        double diagC0 = diag_min(C, p.B);
        double fnA = frob(A), fnC0 = frob(C);
        size_t nNaNC0 = naninf(C);

        // asymétrie (devrait rester ~0 sur une tuile diag)
        double max_asym0 = 0.0;
        for (int j=0;j<p.B;++j) for (int i=0;i<j;++i)
          max_asym0 = std::max(max_asym0, std::abs(C[(size_t)i + (size_t)j*p.B] - C[(size_t)j + (size_t)i*p.B]));

        std::cout << "[WORKER][SYRK]"<<" diag(C)_min=" << diagC0
                  << " ||A||F0=" << fnA
                  << " ||C||F0=" << fnC0
                  << " NaN/Inf A0=" << nNaNC0
                  << " max_asym(C0)=" << max_asym0
                  << std::endl;

        // std::vector<double> C = deserialize_1_block(Cblob);
        // std::vector<double> A = deserialize_1_block(Ablob);

        CHAM_desc_t *dC = nullptr, *dA = nullptr;
        create_desc_1block(&dC, C.data(), p.B);
        create_desc_1block(&dA, A.data(), p.B);

        int info_syrk = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info_syrk = CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        std::cout << "[WORKER][SYRK] : chameleon syrk info " << std::to_string(info_syrk) << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (info_syrk != 0) {CHAMELEON_Desc_Destroy(&dA);
                              CHAMELEON_Desc_Destroy(&dC);
          throw std::runtime_error("[Worker][SYRK] dsyrk info=" + std::to_string(info_syrk));}

        secs   = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops  = 1.0 * (double)p.B * (double)p.B * (double)p.B;
        gflops = flops / (secs * 1e9);

        dump4("[Worker][SYRK] C post", C);
        double diagC1 = diag_min(C, p.B);
        double fnC1 = frob(C);
        size_t nNaNC1 = naninf(C);

        double max_asym1 = 0.0;
        for (int j=0;j<p.B;++j) for (int i=0;i<j;++i)
          max_asym1 = std::max(max_asym1, std::abs(C[(size_t)i + (size_t)j*p.B] - C[(size_t)j + (size_t)i*p.B]));

        std::cout << "[WORKER][SYRK]"<<" diag(C): " << diagC0 << " -> " << diagC1
                  << " ||C||F: " << fnC0 << " -> " << fnC1
                  << " NaN/Inf C: " << nNaNC0 << " -> " << nNaNC1
                  << " max_asym(C): " << max_asym0 << " -> " << max_asym1
                  << " time=" << secs << "s gflops=" << gflops << std::endl;

        absl::string_view out_blockViewC(reinterpret_cast<const char*>(C.data()),C.size() * sizeof(double));
        // const std::string out_blob = serialize_1_block(C);

        CHAMELEON_Desc_Destroy(&dC);
        CHAMELEON_Desc_Destroy(&dA);

        try {
          // taskHandler.send_result(out_id, out_blob).get();
          taskHandler.send_result(out_id, out_blockViewC).get();

        } catch (const std::exception& e) {
          return armonik::api::worker::ProcessStatus(std::string("send_result failed: ") + e.what());
        }
        return armonik::api::worker::ProcessStatus::Ok;     
       }

      // ============================
      // GEMM
      // ============================
      else if (p.op == "GEMM") {

        const auto& expected = taskHandler.getExpectedResults();
        std::cout << " [Worker][GEMM][expected = taskHandler.getExpectedResults()] : expected size ="  << expected.size() << std::endl ;
        const std::string& out_id = expected.front();
        const auto& deps = taskHandler.getDataDependencies();

          auto itC  = deps.find(p.inC);
        if (itC == deps.end())
          return armonik::api::worker::ProcessStatus(" [Worker][GEMM] Missing dependency: " + p.inC);
        const std::string& Cblob  = itC->second;

        auto itAi = deps.find(p.inAi);
        if (itAi == deps.end())
          return armonik::api::worker::ProcessStatus("[Worker][GEMM] Missing dependency: " + p.inAi);
        const std::string& Aiblob = itAi->second;

        auto itAj = deps.find(p.inAj);
        if (itAj == deps.end())
          return armonik::api::worker::ProcessStatus("[Worker][GEMM] Missing dependency: " + p.inAj);
        const std::string& Ajblob = itAj->second;

        const double* ptrC = reinterpret_cast<const double*>(Cblob.data());
        std::vector<double> C(ptrC, ptrC + Cblob.size()/sizeof(double)); 

        const double* ptrAi = reinterpret_cast<const double*>(Aiblob.data());
        std::vector<double> Ai(ptrAi, ptrAi + Aiblob.size()/sizeof(double));

        const double* ptrAj = reinterpret_cast<const double*>(Ajblob.data());
        std::vector<double> Aj(ptrAj, ptrAj + Ajblob.size()/sizeof(double));

        dump4("[Worker][GEMM] C pre",  C);
        dump4("[Worker][GEMM] Ai pre", Ai);
        dump4("[Worker][GEMM] Aj pre", Aj);
        double fnC0 = frob(C), fnAi = frob(Ai), fnAj = frob(Aj);
        size_t nNaNC0 = naninf(C);
        std::cout << "[WORKER][GEMM]"<<" ||C||F: " << fnC0 
                  << " NaN/Inf C: " << nNaNC0 << std::endl;

        // std::vector<double> C  = deserialize_1_block(Cblob);
        // std::vector<double> Ai = deserialize_1_block(Aiblob);
        // std::vector<double> Aj = deserialize_1_block(Ajblob);

        CHAM_desc_t *dC = nullptr, *dAi = nullptr, *dAj = nullptr;
        create_desc_1block(&dC,  C.data(),  p.B);
        create_desc_1block(&dAi, Ai.data(), p.B);
        create_desc_1block(&dAj, Aj.data(), p.B);
        int info_gemm = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        std::cout << "[WORKER][GEMM]] : chameleon gemm info " << std::to_string(info_gemm) << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info_gemm != 0) {CHAMELEON_Desc_Destroy(&dC);
                              CHAMELEON_Desc_Destroy(&dAi);
                              CHAMELEON_Desc_Destroy(&dAj);;
          throw std::runtime_error("[Worker][GEMM] dgemm info=" + std::to_string(info_gemm));}
        secs   = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops  = 2.0 * (double)p.B * (double)p.B * (double)p.B;
        gflops = flops / (secs * 1e9);


        dump4("[GEMM] C post", C);
        double fnC1 = frob(C);
        size_t nNaNC1 = naninf(C);
        std::cout << "[WORKER][GEMM]"<< "||C||F: " << fnC0 << " -> " << fnC1
                  << " (Δ=" << (fnC1 - fnC0) << ")"
                  << " ||Ai||F=" << fnAi << " ||Aj||F=" << fnAj
                  << " NaN/Inf C: " << nNaNC0 << " -> " << nNaNC1
                  << " time=" << secs << "s gflops=" << gflops <<std::endl;

        absl::string_view out_blockViewC(reinterpret_cast<const char*>(C.data()),C.size() * sizeof(double));
        // const std::string out_blob = serialize_1_block(C);

        CHAMELEON_Desc_Destroy(&dC);
        CHAMELEON_Desc_Destroy(&dAi);
        CHAMELEON_Desc_Destroy(&dAj);

          try {
              // taskHandler.send_result(out_id, out_blob).get();
              taskHandler.send_result(out_id, out_blockViewC).get();
            } catch (const std::exception& e) {
              return armonik::api::worker::ProcessStatus(std::string("send_result failed: ") + e.what());
            }
            return armonik::api::worker::ProcessStatus::Ok;
      }
      else {
        return armonik::api::worker::ProcessStatus("Unknown op=" + p.op);
      }

      std::cerr << "[INFO] Execute: end OK\n";
      std::cerr << "[PERF] " 
                << " time=" << secs << " sec"
                << " flops=" << flops << " gflops=" << gflops << "\n";

      return armonik::api::worker::ProcessStatus::Ok;
    } 
    catch (const std::exception& e) {
      return armonik::api::worker::ProcessStatus(std::string("Exception: ") + e.what());
    }
    catch (...) {
      return armonik::api::worker::ProcessStatus("Unknown exception occurred.");
    }
  }
};

int main() {  

  /*
  CHAMELEON_Init(ncpu, ngpu);
  std::cout << "[WORKER]init chameleon ok ";
  std::atexit([]{ CHAMELEON_Finalize(); });
  */

  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::cout << "[WORKER] build=" << __DATE__ << " " << __TIME__ << " cereal=ON\n";

  armonik::api::common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  config.set("ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock");
  config.set("ComputePlane__AgentChannel__Address", "/cache/armonik_agent.sock");

  int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency()));
  int ngpu = env_int("CHM_NGPU",1);
  std::cout << "[WORKER] build2=" << "ncpu=" << " " << ncpu << " & " << " ngpu"<< ngpu; 

  std::cout << "[WORKER] build3=" << "Chameleon initialization\n";
  CHAMELEON_Init(ncpu, ngpu);
  std::cout << "[WORKER] build4=" << "Chameleon initialization sucefful\n"; 


  try {
    armonik::api::worker::WorkerServer::create<DagCholeskyWorker>(config)->run();
  } catch (const std::exception &e) {
  }
  return 0;
}


