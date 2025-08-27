// ============================================================================
//  Client ArmoniK — Cholesky par blocs 
//  Vagues k : POTRF(k) → TRSM(i,k) pour i>k → MAJ(i,j,k) (SYRK/GEMM) pour k<j≤i
//  Chaque tâche lit/écrit des blocs via le store (IDs "blk/i/j").
// ============================================================================

//
#include <string>                       
#include <sstream>                      
#include <vector>                      
#include <iostream>                     
#include "objects.pb.h"                 
#include "utils/Configuration.h"        
#include "logger/logger.h"             
#include "logger/writer.h"              
#include "logger/formatter.h"           
#include "channel/ChannelFactory.h"    
#include "sessions/SessionsClient.h"    
#include "tasks/TasksClient.h"         
#include "results/ResultsClient.h"    
#include "events/EventsClient.h"      
#include <map>                          
#include <random>                   
#include <algorithm>                    


namespace ak_common = armonik::api::common;
namespace ak_client = armonik::api::client;
namespace ak_grpc   = armonik::api::grpc::v1;


// Convention : "blk/<i>/<j>"
//        j=0    j=1    j=2
// i=0  blk/0/0 blk/0/1 blk/0/2
// i=1  blk/1/0 blk/1/1 blk/1/2
// i=2  blk/2/0 blk/2/1 blk/2/2

static std::string blk_id(int i, int j) {
  std::ostringstream oss; oss << "blk/" << i << "/" << j; return oss.str();
}
// Payload POTRF
static std::string make_payload_potrf(int k, int B) {
  std::ostringstream oss;
  oss << "op=POTRF k=" << k << " B=" << B
      << " in="  << blk_id(k,k)     
      << " out=" << blk_id(k,k);    
  return oss.str();
}

// Payload TRSM 
static std::string make_payload_trsm(int i, int k, int B) {
  std::ostringstream oss;
  oss << "op=TRSM i=" << i << " k=" << k << " B=" << B
      << " inL=" << blk_id(k,k)     
      << " inA=" << blk_id(i,k)     
      << " out=" << blk_id(i,k);   
  return oss.str();
}

// Payload SYRK 
static std::string make_payload_syrk(int i, int k, int B) {
  std::ostringstream oss;
  oss << "op=SYRK i=" << i << " k=" << k << " B=" << B
      << " inC=" << blk_id(i,i)     
      << " inA=" << blk_id(i,k)     
      << " out=" << blk_id(i,i);   
  return oss.str();
}


// Payload GEMM 
static std::string make_payload_gemm(int i, int j, int k, int B) {
  std::ostringstream oss;
  oss << "op=GEMM i=" << i << " j=" << j << " k=" << k << " B=" << B
      << " inC="  << blk_id(i,j)    
      << " inAi=" << blk_id(i,k)    
      << " inAj=" << blk_id(j,k)    
      << " out="  << blk_id(i,j);  
  return oss.str();
}

// Génère un bloc B×B aléatoire (POC)
static std::vector<double> gen_random_block(int B, double scale=1.0) {
  static std::mt19937_64 rng(42);            
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> X(B*B);
  for (int t=0;t<B*B;++t) X[t] = dist(rng) * scale;
  return X;
}

// Sérialise un bloc de doubles en chaîne binaire (bytes) pour ResultsStore
static std::string serialize_block(const std::vector<double>& block) {
  return std::string(reinterpret_cast<const char*>(block.data()),
                     block.size()*sizeof(double));
}


int main() {

  ak_common::logger::Logger logger{ak_common::logger::writer_console(),ak_common::logger::formatter_plain(true)};
  ak_common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  logger.info("Initialized client config.");

  ak_client::ChannelFactory channelFactory(config, logger);
  std::shared_ptr<::grpc::Channel> channel = channelFactory.create_channel();

  ak_grpc::TaskOptions taskOptions;
  
  const std::string part_cpu = "cholesky-cpu";
  const std::string part_gpu = "cholesky-gpu";
  const std::string part_hybrid = "cholesky-hybrid"
  const std::string default_partition = part_cpu;

  logger.info("Partitions (allowed in session): cpu=" + part_cpu + ", gpu=" + part_gpu + ", hybrid=" + part_hybrid + ", default=" + default_partition);

  taskOptions.mutable_max_duration()->set_seconds(3600); 
  taskOptions.set_max_retries(3);                        
  taskOptions.set_priority(1);                           
  taskOptions.set_partition_id(default_partition);       

  taskOptions.set_application_name("cholesky-dag");         
  taskOptions.set_application_version("1.0");            
  taskOptions.set_application_namespace("benchmarks");


  ak_client::TasksClient    tasksClient(   ak_grpc::tasks::Tasks::NewStub(channel));
  ak_client::ResultsClient  resultsClient( ak_grpc::results::Results::NewStub(channel));
  ak_client::SessionsClient sessionsClient(ak_grpc::sessions::Sessions::NewStub(channel));
  ak_client::EventsClient   eventsClient(  ak_grpc::events::Events::NewStub(channel));


  const int N  = 10000;                   
  const int B  = 448;                   
  const int Nb = (N + B - 1) / B;          
  logger.info("Problem: N=" + std::to_string(N) + " B=" + std::to_string(B) + " Nb=" + std::to_string(Nb));


  // Les partitions autorisées
  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu, part_gpu, part_hybrid});
  logger.info("Session id = " + session_id);

  // Enregistrer les IDs de résultats pour tous les blocs (triangle inférieur)
  std::vector<std::string> all_block_names;
  all_block_names.reserve(Nb*Nb);
  for (int i=0;i<Nb;++i)
    for (int j=0;j<=i;++j)
      all_block_names.push_back(blk_id(i,j));
  std::map<std::string,std::string> id_map =
      resultsClient.create_results_metadata(session_id, all_block_names);


  //  Soumettre une tâche dans une partition donnée --------------------
  auto submit_partition = [&](const std::string& payload_name,
                                 const std::string& payload_text,
                                 const std::vector<std::string>& out_ids,
                                 const std::string& partition_id)
  {
    // 1) Créer et uploader le payload comme un Result
    auto meta = resultsClient.create_results_metadata(session_id, {payload_name});
    const std::string payload_id = meta[payload_name];
    resultsClient.upload_result_data(session_id, payload_id, payload_text);

    // 2) Construire la TaskCreation + TaskOptions spécifiques à la tâche
    ak_common::TaskCreation tc;
    tc.set_payload_id(payload_id);
    for (const auto& rid : out_ids)
      *tc.add_expected_output_keys() = rid;

    // Copier les options globales puis surcharger la partition pour cette tâche
    ak_grpc::TaskOptions per_task_opts = taskOptions;
    per_task_opts.set_partition_id(partition_id);
    *tc.mutable_task_options() = std::move(per_task_opts);

    // 3) Soumettre
    tasksClient.submit_tasks(session_id, { tc });
  };

  // POC : on génère une matrice : blocs aléatoires, diagonale renforcée.
  logger.info("Uploading initial blocks (lower triangle)...");
  for (int i=0;i<Nb;++i) {
    for (int j=0;j<=i;++j) {
      std::vector<double> block = gen_random_block(B, 0.1);
      if (i==j) {
        // Renforce la diagonale pour rendre positive définie
        for (int d=0; d<B; ++d) block[d*B + d] += (double)B;
      }
      // Upload du bloc sérialisé dans le ResultsStore
      resultsClient.upload_result_data(session_id,
                                       id_map[blk_id(i,j)],
                                       serialize_block(block));
    }
  }
  logger.info("Initial blocks uploaded.");


  // -------------------- Exécution par vagues (k = 0..Nb-1) --------------------
  for (int k=0; k<Nb; ++k) {
    logger.info("Wave k=" + std::to_string(k));

    // POTRF(k,k) : factorise la diagonale (CPU) ----
    {
      std::string pl = make_payload_potrf(k, B);
      std::string payload_name = "payload/potrf/" + std::to_string(k);
      submit_partition(payload_name, pl, { id_map[blk_id(k,k)] }, part_cpu);
      eventsClient.wait_for_result_availability(session_id, { id_map[blk_id(k,k)] });
    }

    // TRSM(i,k) pour i>k : descente de la colonne k (CPU) ----
    std::vector<std::string> trsm_out_ids;   
    for (int i=k+1; i<Nb; ++i) {
      std::string pl = make_payload_trsm(i, k, B);
      std::string payload_name = "payload/trsm/" + std::to_string(i) + "/" + std::to_string(k);
      submit_partition(payload_name, pl, { id_map[blk_id(i,k)] }, part_cpu);
      trsm_out_ids.push_back(id_map[blk_id(i,k)]);
    }
    if (!trsm_out_ids.empty())
      eventsClient.wait_for_result_availability(session_id, trsm_out_ids);

    // MAJ(i,j,k) : SYRK (diag) et GEMM (off-diag)
    // GPU si dispo, sinon CPU fallback 
    const std::string part_for_update = part_gpu.empty() ? part_cpu : part_gpu;
    std::vector<std::string> upd_out_ids;
    for (int i=k+1; i<Nb; ++i) {
      for (int j=k+1; j<=i; ++j) {
        const bool is_diag = (i==j);
        std::string pl = is_diag ? make_payload_syrk(i, k, B)
                                 : make_payload_gemm(i, j, k, B);
        std::string payload_name = std::string("payload/") + (is_diag ? "syrk/" : "gemm/")
                                 + std::to_string(i) + "/" + std::to_string(j)
                                 + "/" + std::to_string(k);
        submit_partition(payload_name, pl, { id_map[blk_id(i,j)] }, part_for_update);
        upd_out_ids.push_back(id_map[blk_id(i,j)]);
      }
    }
    if (!upd_out_ids.empty())
      eventsClient.wait_for_result_availability(session_id, upd_out_ids);

    logger.info("Wave k=" + std::to_string(k) + " done.");
  }

  
  logger.info("All waves completed. L factor is in blk/i/j (i≥j).");
  return 0;
}
