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
#include <map>                          
#include <random>                   
#include <algorithm>  

#include "armonik/common/objects.pb.h"
#include "armonik/common/utils/Configuration.h"
#include "logger/logger.h"             
#include "logger/writer.h"              
#include "logger/formatter.h"           
#include "channel/ChannelFactory.h"    
#include "sessions/SessionsClient.h"    
#include "tasks/TasksClient.h"         
#include "results/ResultsClient.h"    
#include "events/EventsClient.h"      
                  


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
static std::string make_payload_potrf(const std::string& id_in_out, int B) {
  std::ostringstream oss;
  oss << "op=POTRF B=" << B
      << " in=" << id_in_out
      << " out=" << id_in_out;
  return oss.str();
}

// Payload TRSM 
static std::string make_payload_trsm(const std::string& id_Lkk,
                                     const std::string& id_Aik,
                                     int B) {
  std::ostringstream oss;
  oss << "op=TRSM B=" << B
      << " inL=" << id_Lkk
      << " inA=" << id_Aik
      << " out=" << id_Aik;
  return oss.str();
}

// payload SYRK
static std::string make_payload_syrk(const std::string& id_Cii,
                                     const std::string& id_Aik,
                                     int B) {
  std::ostringstream oss;
  oss << "op=SYRK B=" << B
      << " inC=" << id_Cii
      << " inA=" << id_Aik
      << " out=" << id_Cii;
  return oss.str();
}

// Payload GEMM 
static std::string make_payload_gemm(const std::string& id_Cij,
                                     const std::string& id_Aik,
                                     const std::string& id_Ajk,
                                     int B) {
  std::ostringstream oss;
  oss << "op=GEMM B=" << B
  << " inC=" << id_Cij
  << " inAi=" << id_Aik
  << " inAj=" << id_Ajk
  << " out=" << id_Cij;
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
  
  const std::string part_cpu_vm = "cholesky-cpu-vm";
  const std::string part_cpu_aws = "cholesky-cpu-aws";
  const std::string part_gpu_vm = "cholesky-gpu-vm";
  const std::string part_gpu_aws = "cholesky-gpu-aws";
  const std::string part_hybrid_vm = "cholesky-hybrid-vm";
  const std::string part_hybrid_aws = "cholesky-hybrid-aws";
  const std::string default_partition = part_cpu_vm;


logger.info("Partitions (allowed in session): cpu-vm=" + part_cpu_vm + ",cpu-aws=" + part_cpu_aws + ", gpu-vm=" + part_gpu_vm + ", gpu-aws=" + part_gpu_aws + ", hybrid-vm=" + part_hybrid_vm + ", hybrid-aws=" + part_hybrid_aws + ", default=" + default_partition);

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


  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu_vm, part_gpu_vm, part_hybrid_vm});
  logger.info("Session id = " + session_id);

  std::vector<std::string> all_block_names; 
  all_block_names.reserve(Nb*Nb);
  for (int i=0;i<Nb;++i) 
    for (int j=0;j<=i;++j) 
      all_block_names.push_back(blk_name(i,j));
  std::map<std::string,std::string> id_map = resultsClient.create_results_metadata(session_id, all_block_names);

  logger.info("Uploading initial blocks (lower triangle)...");
  for (int i=0;i<Nb;++i) {
    for (int j=0;j<=i;++j) {
      std::vector<double> block = gen_random_block(B, 0.1);
      if (i==j) {
        for (int d=0; d<B; ++d) block[d*B + d] += (double)B;
      }
      resultsClient.upload_result_data(session_id,
                                       id_map[blk_id(i,j)],
                                       serialize_block(block));
    }
  }
  logger.info("Initial blocks uploaded.");

  //  Soumettre une tâche dans une partition donnée --------------------
  auto submit_partition = [&](const std::string& payload_name,
                                 const std::string& payload_text,
                                 const std::vector<std::string>& out_ids,
                                 const std::string& partition_id)
  {
    auto meta = resultsClient.create_results_metadata(session_id, {payload_name});
    const std::string payload_id = meta[payload_name];
    resultsClient.upload_result_data(session_id, payload_id, payload_text);

    ak_common::TaskCreation tc;
    tc.set_payload_id(payload_id);
    for (const auto& rid : out_ids)*tc.add_expected_output_keys() = rid;
    for (const auto& rid : in_ids) *tc.add_data_dependencies()    = rid;

    ak_grpc::TaskOptions per_task_opts = taskOptions;
    per_task_opts.set_partition_id(partition_id);
    *tc.mutable_task_options() = std::move(per_task_opts);

    tasksClient.submit_tasks(session_id, { tc });
  };

  // -------------------- Exécution par vagues (k = 0..Nb-1) --------------------
for (int k=0; k<Nb; ++k) {
  logger.info("Wave k=" + std::to_string(k));


  const std::string id_kk = id_map[blk_name(k,k)];


  // 1) POTRF(k,k) sur CPU
  {
    const std::string pl = make_payload_potrf(id_kk, B);
    const std::string payload_name = "payload/potrf/" + std::to_string(k);
    submit_partition(payload_name, pl, { id_kk }, { id_kk }, part_cpu_vm);
    eventsClient.wait_for_result_availability(session_id, { id_kk });
  }


  // 2) TRSM(i,k) pour i>k sur CPU
  std::vector<std::string> trsm_out_ids;
  for (int i=k+1; i<Nb; ++i) {
    const std::string id_ik = id_map[blk_name(i,k)];
    const std::string pl = make_payload_trsm(id_kk, id_ik, B);
    const std::string payload_name = "payload/trsm/" + std::to_string(i) + "/" + std::to_string(k);
    submit_partition(payload_name, pl, { id_ik }, { id_kk, id_ik }, part_cpu_vm);
    trsm_out_ids.push_back(id_ik);
  }
  if (!trsm_out_ids.empty())
    eventsClient.wait_for_result_availability(session_id, trsm_out_ids);


  // 3) MAJ(i,j,k) : SYRK (diag) / GEMM (hors-diag) — GPU si dispo
  const std::string part_for_update = part_gpu.empty() ? part_cpu_vm : part_gpu_vm;
  std::vector<std::string> upd_out_ids;
  for (int i=k+1; i<Nb; ++i) {
    const std::string id_ik = id_map[blk_name(i,k)];
    for (int j=k+1; j<=i; ++j) {
      if (i == j) {
        const std::string id_ii = id_map[blk_name(i,i)];
        const std::string pl = make_payload_syrk(id_ii, id_ik, B);
        const std::string payload_name = std::string("payload/syrk/") + std::to_string(i) + "/" + std::to_string(k);
        submit_partition(payload_name, pl, { id_ii }, { id_ii, id_ik }, part_for_update);
        upd_out_ids.push_back(id_ii);
      } else {
        const std::string id_ij = id_map[blk_name(i,j)];
        const std::string id_jk = id_map[blk_name(j,k)];
        const std::string pl = make_payload_gemm(id_ij, id_ik, id_jk, B);
        const std::string payload_name = std::string("payload/gemm/") + std::to_string(i) + "/" + std::to_string(j) + "/" + std::to_string(k);
        submit_partition(payload_name, pl, { id_ij }, { id_ij, id_ik, id_jk }, part_for_update);
        upd_out_ids.push_back(id_ij);
      }
    }
  }
  if (!upd_out_ids.empty())
    eventsClient.wait_for_result_availability(session_id, upd_out_ids);


logger.info("Wave k=" + std::to_string(k) + " done.");
}


logger.info("All waves completed. L factor is in blk/i/j (i≥j) — IDs carry final data.");
return 0;
}