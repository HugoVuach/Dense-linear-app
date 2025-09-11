
#include <string>                       
#include <sstream>                      
#include <vector>                      
#include <iostream> 
#include <map>                          
#include <random>                   
#include <algorithm>  
#include <unordered_map>
#include <cstring>
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
                  


namespace ak_common = armonik::api::common;
namespace ak_client = armonik::api::client;
namespace ak_grpc   = armonik::api::grpc::v1;

// ============================================================================
// Payloads creators
// ============================================================================

static std::string make_payload_potrf(const std::string& id_in, int B) {
  std::ostringstream oss;
  oss << "op=POTRF B=" << B << " in=" << id_in;
  return oss.str();
}
static std::string make_payload_trsm(const std::string& id_Lkk,
                                     const std::string& id_Aik,
                                     int B) {
  std::ostringstream oss;
  oss << "op=TRSM B=" << B << " inL=" << id_Lkk << " inA=" << id_Aik;
  return oss.str();
}
static std::string make_payload_syrk(const std::string& id_Cii,
                                     const std::string& id_Aik,
                                     int B) {
  std::ostringstream oss;
  oss << "op=SYRK B=" << B << " inC=" << id_Cii << " inA=" << id_Aik;
  return oss.str();
}
static std::string make_payload_gemm(const std::string& id_Cij,
                                     const std::string& id_Aik,
                                     const std::string& id_Ajk,
                                     int B) {
  std::ostringstream oss;
  oss << "op=GEMM B=" << B
      << " inC=" << id_Cij << " inAi=" << id_Aik << " inAj=" << id_Ajk;
  return oss.str();
}

// ===========================================================================
// Generate random blocks
// ==========================================================================

// 1. Generate a random block
static std::vector<double> generate_random_block(int B, double scale=1.0) {
  static std::mt19937_64 rng(42);            
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> X(B*B);
  for (int t=0;t<B*B;++t) X[t] = dist(rng) * scale;
  return X;
}

// ============================================================================
// Serialize and Deserialize function
// ============================================================================

// 1. Serialize function
static std::string serialize_block(const std::vector<double>& block) {
  return std::string(reinterpret_cast<const char*>(block.data()),
                     block.size()*sizeof(double));
}

// 2. Deserialize function
static std::vector<double> deserialize_block(const std::string& blob, int B) {
  const size_t need = static_cast<size_t>(B) * static_cast<size_t>(B) * sizeof(double);
  if (blob.size() != need) throw std::runtime_error("Bad block size");
  std::vector<double> v(B*B);
  std::memcpy(v.data(), blob.data(), need);
  return v;
}

// ============================================================================
// Bloc ID
// ============================================================================
// 1. Block ID from (i,j)
//
//  block_id(2, 1) → "blk/2/1"
// 
static std::string block_id(int i, int j) {
  std::ostringstream oss; oss << "blk/" << i << "/" << j; return oss.str();
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
  logger.info("Partitions (allowed in session): cpu-vm=" + part_cpu_vm);
  taskOptions.mutable_max_duration()->set_seconds(3600); 
  taskOptions.set_max_retries(3);                        
  taskOptions.set_priority(1);                           
  taskOptions.set_partition_id(part_cpu_vm);       
  taskOptions.set_application_name("cholesky-dag");         
  taskOptions.set_application_version("1.0");            
  taskOptions.set_application_namespace("benchmarks");
  ak_client::TasksClient    tasksClient(   ak_grpc::tasks::Tasks::NewStub(channel));
  ak_client::ResultsClient  resultsClient( ak_grpc::results::Results::NewStub(channel));
  ak_client::SessionsClient sessionsClient(ak_grpc::sessions::Sessions::NewStub(channel));
  ak_client::EventsClient   eventsClient(  ak_grpc::events::Events::NewStub(channel));
  const int N  = 12;                   
  const int B  = 4;                   
  const int Nb = (N + B - 1) / B;
  logger.info("Problem: N=" + std::to_string(N) +" B=" + std::to_string(B) +" Nb=" + std::to_string(Nb));          
  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu_vm});





  //                                                      ["blk/0/0",
  // lower_triangle_block_keys.push_back(block_id(i,j))    "blk/1/0", "blk/1/1",
  //                                                       "blk/2/0", "blk/2/1", "blk/2/2"]
  std::vector<std::string> lower_triangle_block_keys; 
  lower_triangle_block_keys.reserve(Nb*Nb);
  for (int i=0;i<Nb;++i) 
    for (int j=0;j<=i;++j) 
      lower_triangle_block_keys.push_back(block_id(i,j));


  // IDs init (lecture seule ensuite)
  // id_map = resultsClient.create_results_metadata(session_id, lower_triangle_block_keys) 
  //                →  {"blk/0/0": "r_7adf31",
  //                    "blk/1/0": "r_c1f902",
  //                    "blk/1/1": "r_b0a513",
  //                    "blk/2/0": "r_83e410",
  //                    "blk/2/1": "r_9912aa",
  //                    "blk/2/2": "r_0dce77"}

  std::map<std::string,std::string> id_map = resultsClient.create_results_metadata(session_id, lower_triangle_block_keys);

  logger.info("Uploading initial blocks (lower triangle)...");
  for (int i=0;i<Nb;++i) {
    for (int j=0;j<=i;++j) {
      std::vector<double> block = generate_random_block(B, 0.1);
      if (i==j) {
        for (int d=0; d<B; ++d) block[d*B + d] += (double)B;
      }
      resultsClient.upload_result_data(session_id,id_map[block_id(i,j)],serialize_block(block));
    }
  }
  logger.info("Initial blocks uploaded.");




auto force_kv = [](std::string s, const std::string& key, const std::string& val){
  const std::string pat = key + "=";
  auto p = s.find(pat);
  if (p == std::string::npos) return s + " " + key + "=" + val;
  auto q = p + pat.size();
  auto r = s.find(' ', q);
  return s.replace(q, (r==std::string::npos ? s.size()-q : r-q), val);
};


  std::unordered_map<std::string, std::string> latest{id_map.begin(), id_map.end()};

  auto submit_one = [&](const std::string& payload_text_base,
                              std::vector<std::string> data_deps,
                        const std::string& partition_id,
                              std::string& output_id) {

    std::map<std::string, std::string> result_ids = resultsClient.create_results_metadata(session_id, {"output", "payload"});
    output_id                     = result_ids["output"];
    const std::string payload_id  = result_ids["payload"];


    logger.info("[SUBMIT1] new ids payload_id=" + payload_id + " output_id=" + output_id);

    std::sort(data_deps.begin(), data_deps.end());
    data_deps.erase(std::unique(data_deps.begin(), data_deps.end()), data_deps.end());

    std::string payload_text = force_kv(payload_text_base, "out", output_id);

    logger.info("[SUBMIT2] payload='" + payload_text + "'");
    std::ostringstream dep_stream;
    for (auto &d : data_deps) dep_stream << d << " ";
    logger.info("[SUBMIT3] data_deps=" + dep_stream.str());

    resultsClient.upload_result_data(session_id, payload_id, payload_text);


    ak_common::TaskCreation tc;
    tc.payload_id           = payload_id;
    tc.expected_output_keys = {output_id};
    tc.data_dependencies    = std::move(data_deps);


    ak_grpc::TaskOptions per_task_opts = taskOptions;
    per_task_opts.set_partition_id(partition_id);
    tasksClient.submit_tasks(session_id, {tc}, per_task_opts);

    logger.info("[SUBMIT4] task submitted.");

    eventsClient.wait_for_result_availability(session_id, {output_id});
    logger.info("[WAIT] output ready: " + output_id);

  };

  // ------------------------ Exécution par vagues k -------------------------------
  for (int k = 0; k < Nb; ++k) {
    logger.info("Wave k=" + std::to_string(k));

    // 1) POTRF(k,k)
    {
      const std::string Lkk_in = latest.at(block_id(k, k));
      std::string Lkk_out;
      const std::string payload = make_payload_potrf(Lkk_in, B); 
      submit_one(payload, {Lkk_in}, part_cpu_vm, Lkk_out);
      latest[block_id(k, k)] = Lkk_out; 
    }

    // 2) TRSM(i,k) pour i>k
    {
      std::vector<std::string> trsm_outs; trsm_outs.reserve(std::max(0, Nb - (k + 1)));
      for (int i = k + 1; i < Nb; ++i) {
        const std::string Lkk = latest.at(block_id(k, k));
        const std::string Aik_in = latest.at(block_id(i, k));
        std::string Aik_out;
        const std::string payload = make_payload_trsm(Lkk, Aik_in, B);
        submit_one(payload, {Lkk, Aik_in}, part_cpu_vm, Aik_out);
        latest[block_id(i, k)] = Aik_out;
        trsm_outs.push_back(Aik_out);
      }
      (void)trsm_outs;
    }

    // 3) MAJ(i,j,k) : SYRK (diag) et GEMM (hors-diag)
    {
      const std::string part_for_update = part_cpu_vm; 
      for (int i = k + 1; i < Nb; ++i) {
        const std::string Aik = latest.at(block_id(i, k));
        for (int j = k + 1; j <= i; ++j) {
          if (i == j) {
            const std::string Cii_in = latest.at(block_id(i, i));
            std::string Cii_out;
            const std::string payload = make_payload_syrk(Cii_in, Aik, B);
            submit_one(payload, {Cii_in, Aik}, part_for_update, Cii_out);
            latest[block_id(i, i)] = Cii_out;
          } else {
            const std::string Cij_in = latest.at(block_id(i, j));
            const std::string Ajk    = latest.at(block_id(j, k));
            std::string Cij_out;
            const std::string payload = make_payload_gemm(Cij_in, Aik, Ajk, B);
            submit_one(payload, {Cij_in, Aik, Ajk}, part_for_update, Cij_out);
            latest[block_id(i, j)] = Cij_out;
          }
        }
      }
    }

    logger.info("Wave k=" + std::to_string(k) + " done.");
  }

  logger.info("All waves completed. L factor is in blk/i/j (i≥j) — latest[...] holds final IDs.");
  return 0;
}