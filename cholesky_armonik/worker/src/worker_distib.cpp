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
#include <grpcpp/grpcpp.h>                 
#include "grpcpp/support/sync_stream.h"    
#include "objects.pb.h"                     
#include "utils/WorkerServer.h"             
#include "Worker/ArmoniKWorker.h"          
#include "Worker/ProcessStatus.h"           
#include "Worker/TaskHandler.h"             
#include "exceptions/ArmoniKApiException.h" 

extern "C" { 
#include <chameleon.h>   
}



// Lecture d'une variable d'environnement et conversion en entier positif.
// defval, valeur par défaut à utiliser si la variable n'existe pas
static int env_int(const char* key, int defval) {
  if (const char* s = std::getenv(key)) {      
    try { 
      return std::max(0, std::stoi(s));        
    } catch (...) {
    }
  }
  return defval;                               
}

// Retourne la date et l'heure actuelles sous forme d'une chaîne formatée.
static std::string now_ts() {
  std::time_t t = std::time(nullptr);                          
  char buf[64];                                                
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S",         
                std::localtime(&t));                          
  return std::string(buf); 
}

// Parse une payload "k=v k=v ..." en map<string,string>
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
// recherche une clé dans une map de chaînes et retourne sa valeur convertie en entier, ou lève une exception si la clé est absente
static int to_int(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return std::stoi(it->second);
}

// recherche une clé dans une map de chaînes et retourne sa valeur en std::string, ou lève une exception si la clé est absente.
static std::string to_str(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return it->second;
}

// Désérialise un bloc blob binaire -> doubles   
static std::vector<double> deserialize_block(const std::string& blob, int B) {
  const size_t need = static_cast<size_t>(B) * static_cast<size_t>(B) * sizeof(double);
  if (blob.size() != need)
    throw std::runtime_error("Invalid block size: expected " + std::to_string(need) + " bytes, got " + std::to_string(blob.size()));
  std::vector<double> v(B*B);
  std::memcpy(v.data(), blob.data(), need);
  return v;
}

// Sérialise un bloc  doubles -> blob binaire
static std::string serialize_block(const std::vector<double>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

// télécharge et retourne le contenu d’un résultat identifié par resultId en appelant get_result(...).get() sur le TaskHandler
static std::string download_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId) {
  return th.get_result(resultId).get();
}

// Cette fonction envoie (uploade) des données associées à un resultId via le TaskHandler, et attend la fin de l’opération avec .get()
static void upload_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId, const std::string& data) {
  th.send_result(resultId, data).get();
}

// Cette fonction crée un descripteur CHAMELEON correspondant à une matrice carrée de taille B×B stockée dans le buffer ptr, en initialisant un seul bloc (1×1) de taille B
static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
}



class DagCholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent): ArmoniKWorker(std::move(agent)) {}

  // Execute() est appelée pour chaque tâche que le Compute Plane assigne à ce worker
  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    try {
      // 1) Récupérer et parser la payload k=v 
      const std::string payload = taskHandler.getPayload();
      auto kv = parse_kv_payload(payload);
      const std::string op = to_str(kv, "op");  
      const int B = to_int(kv, "B");     


      // extraire les identifiants des entrées/sorties selon l’opération BLAS/Cholesky demandée
      // = dispatcher des dépendances
      //  - POTRF : in=blk/k/k → out=blk/k/k
      //  - TRSM  : inL=blk/k/k, inA=blk/i/k → out=blk/i/k
      //  - SYRK  : inC=blk/i/i, inA=blk/i/k → out=blk/i/i
      //  - GEMM  : inC=blk/i/j, inAi=blk/i/k, inAj=blk/j/k → out=blk/i/j
      std::string in, inL, inA, inC, inAi, inAj, out;
      if      (op == "POTRF") { in  = to_str(kv, "in");  out = to_str(kv, "out"); }
      else if (op == "TRSM")  { inL = to_str(kv,"inL");  inA = to_str(kv,"inA");  out= to_str(kv,"out"); }
      else if (op == "SYRK")  { inC = to_str(kv,"inC");  inA = to_str(kv,"inA");  out= to_str(kv,"out"); }
      else if (op == "GEMM")  { inC = to_str(kv,"inC");  inAi= to_str(kv,"inAi"); inAj= to_str(kv,"inAj"); out= to_str(kv,"out"); }
      else return armonik::api::worker::ProcessStatus("Unknown op=" + op);

      int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency())); 
      int ngpu = env_int("CHM_NGPU", 0);                                                       

      CHAMELEON_Init(ncpu, ngpu);
      int info = 0; 
      std::vector<double> A(B*B), L(B*B), C(B*B), Ai(B*B), Aj(B*B);
      CHAM_desc_t *dA=nullptr, *dL=nullptr, *dC=nullptr, *dAi=nullptr, *dAj=nullptr;

      // Téléchargement des blocs d'entrée (depuis ResultsStore) + création des desc
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
        C  = deserialize_block(download_blob(taskHandler, inC),  B);
        Ai = deserialize_block(download_blob(taskHandler, inAi), B);
        Aj = deserialize_block(download_blob(taskHandler, inAj), B);
        create_desc_1block(&dC,  C.data(),  B);
        create_desc_1block(&dAi, Ai.data(), B);
        create_desc_1block(&dAj, Aj.data(), B);
      }

      // Le calcul est différent pour chaque méthode 
      //     - POTRF(B) : 1/3 B*3
      //     - PTRSM(B×B, Right, triangular B) : 1/2 B*3
      //     - SYRK(B,B) : B*3
      //     - GEMM(B,B,B) : 2 B*3
      struct timespec t0{}, t1{};
      auto Bd = static_cast<double>(B);  
      double secs = 0.0;
      double flops = 0.0;                
      double gflops = 0.0;               

      
      if (op == "POTRF") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) throw std::runtime_error("dpotrf info=" + std::to_string(info));
        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * Bd*Bd*Bd;         
        gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(A)); 
      }

      else if (op == "TRSM") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 0.5 * Bd*Bd*Bd;              
        gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(A)); 
      }

      else if (op == "SYRK") {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = Bd*Bd*Bd;                     
        gflops = flops / (secs * 1e9);
        upload_blob(taskHandler, out, serialize_block(C)); 
      }

      else { 
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 2.0 * Bd*Bd*Bd;               
        gflops = flops / (secs * 1e9);       
        upload_blob(taskHandler, out, serialize_block(C)); 
      }


      if (dA)  CHAMELEON_Desc_Destroy(&dA);
      if (dL)  CHAMELEON_Desc_Destroy(&dL);
      if (dC)  CHAMELEON_Desc_Destroy(&dC);
      if (dAi) CHAMELEON_Desc_Destroy(&dAi);
      if (dAj) CHAMELEON_Desc_Destroy(&dAj);
      CHAMELEON_Finalize(); 

      return armonik::api::worker::ProcessStatus::Ok;
    }
    catch (const std::exception& e) {
      std::cerr << "[DagWorker] ERROR: " << e.what() << "\n";
      return armonik::api::worker::ProcessStatus(e.what());
    }
  }
};


int main() {
  std::cout << "DagCholeskyWorker started. gRPC version = " << grpc::Version() << "\n";
  armonik::api::common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  config.set("ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock"); // // socket Worker
  config.set("ComputePlane__AgentChannel__Address",  "/cache/armonik_agent.sock");  // // socket Agent

  try {
    armonik::api::worker::WorkerServer::create<DagCholeskyWorker>(config)->run();
  } catch (const std::exception &e) {
    std::cerr << "Error in worker: " << e.what() << std::endl;
  }

  std::cout << "Stopping Server..." << std::endl;
  return 0;
}
