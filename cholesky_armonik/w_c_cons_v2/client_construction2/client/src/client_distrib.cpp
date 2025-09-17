
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
#include "armonik/client/channel/ChannelFactory.h"    
#include "armonik/client/sessions/SessionsClient.h" 
#include "armonik/client/tasks/TasksClient.h"         
#include "armonik/client/results/ResultsClient.h"    
#include "armonik/client/events/EventsClient.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "absl/strings/string_view.h"
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"
#include <sstream>
namespace ak_common = armonik::api::common;
namespace ak_client = armonik::api::client;
namespace ak_grpc   = armonik::api::grpc::v1;


/*
Entrées :
id_Cij = "5c60f693-bef5-e011-a485-80ee7300c695", id_Aik = "7f3a2c1d-8b9e-4a0b-9a7c-1e2d3c4b5a6f", id_Ajk = "9a7c1e2d-3c4b-5a6f-7f3a-2c1d8b9e4a0b", B = 448

Sortie (JSON) :
{"op":"GEMM","B":448,
"inC":"5c60f693-bef5-e011-a485-80ee7300c695",
"inAi":"7f3a2c1d-8b9e-4a0b-9a7c-1e2d3c4b5a6f",
"inAj":"9a7c1e2d-3c4b-5a6f-7f3a-2c1d8b9e4a0b"}
*/

static std::string make_payload_potrf(const std::string& id_in, int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op"); w.String("POTRF");
  w.Key("B");  w.Int(B);
  w.Key("in"); w.String(id_in.c_str(), (rapidjson::SizeType)id_in.size());
  w.EndObject();
  return sb.GetString();
}

static std::string make_payload_trsm(const std::string& id_Lkk,
                                     const std::string& id_Aik,
                                     int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op");  w.String("TRSM");
  w.Key("B");   w.Int(B);
  w.Key("inL"); w.String(id_Lkk.c_str(), (rapidjson::SizeType)id_Lkk.size());
  w.Key("inA"); w.String(id_Aik.c_str(), (rapidjson::SizeType)id_Aik.size());
  w.EndObject();
  return sb.GetString();
}

static std::string make_payload_syrk(const std::string& id_Cii,
                                     const std::string& id_Aik,
                                     int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op");  w.String("SYRK");
  w.Key("B");   w.Int(B);
  w.Key("inC"); w.String(id_Cii.c_str(), (rapidjson::SizeType)id_Cii.size());
  w.Key("inA"); w.String(id_Aik.c_str(), (rapidjson::SizeType)id_Aik.size());
  w.EndObject();
  return sb.GetString();
}

static std::string make_payload_gemm(const std::string& id_Cij,
                                     const std::string& id_Aik,
                                     const std::string& id_Ajk,
                                     int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op");   w.String("GEMM");
  w.Key("B");    w.Int(B);
  w.Key("inC");  w.String(id_Cij.c_str(), (rapidjson::SizeType)id_Cij.size());
  w.Key("inAi"); w.String(id_Aik.c_str(), (rapidjson::SizeType)id_Aik.size());
  w.Key("inAj"); w.String(id_Ajk.c_str(), (rapidjson::SizeType)id_Ajk.size());
  w.EndObject();
  return sb.GetString();
}




static std::vector<double> generate_random_B_block(int B, double scale=1.0) {
  static std::mt19937_64 rng(42);            
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> X(B*B);
  for (int t=0;t<B*B;++t) X[t] = dist(rng) * scale;
  return X;
}


//  Prend en entrée deux entiers i et j et retourne une chaîne de caractères
// représentant l'identifiant du bloc au format "blk/i/j".
// ex : block_id(2, 1) → "blk/2/1"
static std::string block_id_from_ij(int i, int j) {
  std::ostringstream oss; oss << "blk/" << i << "/" << j; return oss.str();
}


int main() {

  ak_common::logger::Logger logger{ak_common::logger::writer_console(),ak_common::logger::formatter_plain(true)};
  ak_common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  ak_client::ChannelFactory channelFactory(config, logger);
  std::shared_ptr<::grpc::Channel> channel = channelFactory.create_channel();
  ak_grpc::TaskOptions taskOptions;
  const std::string part_cpu_vm = "cholesky-cpu-vm";
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
  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu_vm});


// =========================================================================================================================================================
// Etape : création des ID blk/i/j et de leur ID associé
// =========================================================================================================================================================
  //                                                         ["blk/0/0", 
  // lower_triangle_block_keys.push_back(block_id(i,j))  ->   "blk/1/0", "blk/1/1",
  //                                                          "blk/2/0", "blk/2/1", "blk/2/2"]
  //
  // id_map →  {"blk/0/0": "15a98827-0053-47d8-86b8-9e0ef7250685",
  //                                [...]
  //            "blk/2/1": "db20d77b-524d-4f49-8e26-7551c54199b0"}

  std::vector<std::string> lower_triangle_block_keys; 
  lower_triangle_block_keys.reserve(Nb*(Nb+1)/2);
  for (int i=0;i<Nb;++i) 
    for (int j=0;j<=i;++j)                                          
      lower_triangle_block_keys.push_back(block_id_from_ij(i,j));   

  std::map<std::string,std::string> id_map = resultsClient.create_results_metadata(session_id, lower_triangle_block_keys); 
  std::cout << "[CLIENT][id_map initialisation] id_map size=" << id_map.size() << std::endl;
  for (const auto& [key, val] : id_map) {
    std::cout << "  " << key << " -> " << val << std::endl;
  }                                                                                                                        

// =========================================================================================================================================================
// Etape : création des blocs du triangle inférieur (diag comprise) + upload_result_data avec une ID initial propre à chaque block crée à l'étape precedente
// =========================================================================================================================================================

  // Pour chaque block std::vector<double> que l'on génère on les converti en absl::string_view  blockView
  // 
  //     * block
  //           - block = generate_random_B_block()
  //           - block.size() = 16 = 4 * 4 = B * B
  //
  //     * bytes
  //           - bytes = block.size() * sizeof(double) = 16 * 8 = 128
  //
  //     * blockView
  //           - conversion obligatoire pour respecter le format de upload_result()
  //           - blockView(reinterpret_cast<const char*>(block.data()),block.size()*sizeof(double))
  //             blockView(reinterpret_cast<const char*>(block.data()),128)
  //           - blockView.size() = 128 = block.size() * sizeof(double)

  for (int i=0;i<Nb;++i) {
    for (int j=0;j<=i;++j) {
      std::vector<double> block = generate_random_B_block(B, 0.1);
      if (i==j) {
        for (int d=0; d<B; ++d) block[d*B + d] += (double)B;
      }                                                                                                                
      auto bytes = block.size() * sizeof(double);                                                                      // Pour chaque block 
      std::cout << "[CLIENT][block = generate_random_B_block()] key=" << block_id_from_ij(i,j) << std::endl;           //            key = blk/i/j
      std::cout << "[CLIENT][block = generate_random_B_block()] block size =" << block.size() << std::endl;            //     block size = 16
      std::cout << "[CLIENT][block = generate_random_B_block()] byte =" << block.size()*sizeof(double) << std::endl;   //           byte = 128
      absl::string_view  blockView(reinterpret_cast<const char*>(block.data()),block.size()*sizeof(double));
      std::cout << "[CLIENT][blockView] blockView.size =" << blockView.size() << std::endl;                            // blockView.size = 128
      const std::string& result_id = id_map.at(block_id_from_ij(i,j));
      std::cout << "[CLIENT][result_id id_map] result_id =" << result_id << std::endl;                                 // c'est le bon result ID qui est associé à la clé blk/i/j
      resultsClient.upload_result_data(session_id, result_id, blockView);                                              // upload initial de chaque block

    }
  }

// =========================================================================================================================================================
// Etape : summit_one
// =========================================================================================================================================================

//
// results["output"] → c’est l’ID de résultat où le worker doit écrire la sortie de la tâche (celui dans expected_output_keys).
// results["payload"] → c’est l’ID de résultat qui contient l’entrée de la tâche.

std::unordered_map<std::string, std::string> latest{id_map.begin(), id_map.end()};
std::cout << "[CLIENT][LATEST] latest size=" << latest.size() << std::endl;
for (const auto& [key, val] : latest) {
  std::cout << " [CLIENT][LATEST] latest key / value " << key << " -> " << val << std::endl;
}

// Soumet une tâche : payload_json = entrée, data_deps = IDs des résultats requis,
// partition_id = partition ArmoniK, output_id = rempli avec le nouvel ID de sortie créé.
// submit_one :
//        -  crée les deux résultats (payload pour l’entrée, output pour la sortie),
//        -  uploade le payload JSON,
//        -  soumet la tâche avec ses dépendances,
//        -  attend que la sortie soit disponible,
//        -  et retourne l’ID de sortie (output_id).
// 
// ajouter un download_result_data(session_id, Lkk_out) uniquement si pour inspecter le contenu

auto submit_one = [&](const std::string& payload_json,
                      std::vector<std::string> data_deps,
                      const std::string& partition_id ) {  
  
  std::cout << " [CLIENT][SUBMIT_ONE] : payload_json = " << payload_json << std::endl;                      // payload_json = {"op":"POTRF","B":4,"in":"15a98827-0053-47d8-86b8-9e0ef7250685"} qui a le même in que dans id_map
  std::cout << " [CLIENT][SUBMIT_ONE] : payload_json_len = " << payload_json.size() << std::endl;           // payload_json_len = 64
  std::cout << " [CLIENT][SUBMIT_ONE] : data_deps.size() = " << data_deps.size() << std::endl;              // data_deps.size() = 1 Normal pour POTRF
  std::cout << " [CLIENT][SUBMIT_ONE] : partition_id = " << partition_id << std::endl;                      // partition_id = cholesky-cpu-vm
                 
              
  // 1) Pour chaque tâche on recréer des Id de sortie car on ne peux pas réecrire le resultat sur le meme Id d'entrée : in 
  //  Crée deux résultats : un pour l'entrée ("payload"), un pour la sortie ("output")
  const auto result_ids = resultsClient.create_results_metadata(session_id, {"output", "payload"});         //
  const std::string output_id   = result_ids.at("output");
  const std::string& payload_id = result_ids.at("payload");
  std::cout <<  " [CLIENT][SUBMIT_ONE] : output_id = " << output_id << std::endl;                           //  output_id = 5d995457-3d16-4c16-be20-90deae7e02ae # c'est nouveau 
  std::cout <<  " [CLIENT][SUBMIT_ONE] : payload_id = " << payload_id << std::endl;                         // payload_id = ef13b5d9-2c3a-44ba-b026-5caeeb757485 # c'est nouveau

  // 2) Déduplique les dépendances ?
  std::sort(data_deps.begin(), data_deps.end());
  data_deps.erase(std::unique(data_deps.begin(), data_deps.end()), data_deps.end());
  std::cout << "[CLIENT][SUBMIT_ONE] : deps_ids=[";                                                         // deps_ids=[15a98827-0053-47d8-86b8-9e0ef7250685] qui est le meme que le in de payload_json
  for (size_t i = 0; i < data_deps.size(); ++i) {
    std::cout << (i ? "," : "") << data_deps[i];
  }
  std::cout << "]" << std::endl;

  resultsClient.upload_result_data(session_id, payload_id, payload_json);

  // 4) Soumission de la tâche : le worker écrira dans expected_output_keys[0]
  ak_common::TaskCreation tc;
  tc.payload_id           = payload_id;
  tc.expected_output_keys = { output_id };
  tc.data_dependencies    = std::move(data_deps);

  ak_grpc::TaskOptions per_task_opts = taskOptions;
  per_task_opts.set_partition_id(partition_id);

  std::cout << "[CLIENT][SUBMIT_ONE] : submit_tasks... expected_output=" << output_id << std::endl;
  tasksClient.submit_tasks(session_id, { tc }, per_task_opts);
  eventsClient.wait_for_result_availability(session_id, { output_id });

  std::cout << "[CLIENT][SUBMIT_ONE] : output available: " << output_id << std::endl;
  return output_id;
};

  // ------------------------ Exécution par vagues k -------------------------------
  for (int k = 0; k < Nb; ++k) {
    logger.info("Wave k=" + std::to_string(k));

    // 1) POTRF(k,k)
    {
      const std::string Lkk_in = latest.at(block_id_from_ij(k, k));                                                  // Lkk_in =15a98827-0053-47d8-86b8-9e0ef7250685 
      const std::string payload = make_payload_potrf(Lkk_in, B); 
        std::cout << " [CLIENT][POTRF(k,k)] : blk" << block_id_from_ij(k, k) << std::endl;
        std::cout << " [CLIENT][POTRF(k,k)] :  Lkk_in =" << Lkk_in << std::endl;
        std::cout << " [CLIENT][POTRF(k,k)] : payload_len=" << payload.size() << std::endl;
        std::cout << " [CLIENT][POTRF(k,k)] :  payload_preview=" << payload.substr(0, 120)  << std::endl;

      const std::string Lkk_out = submit_one(payload, {Lkk_in}, part_cpu_vm);
      std::cout << " [CLIENT][POTRF(k,k)] : Lkk_out=" << Lkk_out  << std::endl;

      latest[block_id_from_ij(k, k)] = Lkk_out;
    }

    // 2) TRSM(i,k) pour i>k
    {
      const std::string Lkk_in = latest.at(block_id_from_ij(k, k));
      for (int i = k + 1; i < Nb; ++i) {
        const std::string Aik_in  = latest.at(block_id_from_ij(i, k));
        const std::string payload = make_payload_trsm(Lkk_in, Aik_in, B);
        const std::string Aik_out = submit_one(payload, {Lkk_in, Aik_in}, part_cpu_vm);
        latest[block_id_from_ij(i, k)] = Aik_out;
      }
    }

    // 3) MAJ(i,j,k) : SYRK (diag) et GEMM (hors-diag)
    {
      const std::string part_for_update = part_cpu_vm; 

      for (int i = k + 1; i < Nb; ++i) {
        const std::string Aik_in = latest.at(block_id_from_ij(i, k));

        for (int j = k + 1; j <= i; ++j) {
          if (i == j) {
            const std::string Cii_in  = latest.at(block_id_from_ij(i, i));
            const std::string payload = make_payload_syrk(Cii_in, Aik_in, B);
            const std::string Cii_out = submit_one(payload, {Cii_in, Aik_in}, part_for_update);
            latest[block_id_from_ij(i, i)] = Cii_out;

          } else {
            const std::string Cij_in = latest.at(block_id_from_ij(i, j));
            const std::string Ajk_in = latest.at(block_id_from_ij(j, k));
            const std::string payload = make_payload_gemm(Cij_in, Aik_in, Ajk_in, B);
            const std::string Cij_out = submit_one(payload, {Cij_in, Aik_in, Ajk_in}, part_for_update);
            latest[block_id_from_ij(i, j)] = Cij_out;
          }
        }
      }
    }

    logger.info("Wave k=" + std::to_string(k) + " done.");
  }

  logger.info("All waves completed.");
  return 0;
}