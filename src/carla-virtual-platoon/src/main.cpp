#include "shared_carlalib.h"
#include "CarlaLocation.hpp"
#include "FrontCamera.hpp"
#include "FrontLidar.hpp"
#include "TruckControl.hpp"
#include "TruckStatus.hpp"
#include <thread>
#include <unistd.h>
#include <sys/wait.h>

std::string host = "localhost";
uint16_t port = 2000u;
cc::Client *client;
cc::World *world;
carla::SharedPtr<carla::client::BlueprintLibrary> blueprint_library;

/// Pick a random element from @a range.
template <typename RangeT, typename RNG>
static auto &RandomChoice(const RangeT &range, RNG &&generator) {
  EXPECT_TRUE(range.size() > 0u);
  std::uniform_int_distribution<size_t> dist{0u, range.size() - 1u};
  return range[dist(std::forward<RNG>(generator))];
}

carla::geom::Location GetTruckLocation(int truck_num, std::string map_name) {
    float x = 0.0f , y = 0.0f ,z =0.0f;
    if (truckLocations.find(map_name) != truckLocations.end() && truckLocations[map_name].find(truck_num) != truckLocations[map_name].end()) {
        x = truckLocations[map_name][truck_num][0];
        y = truckLocations[map_name][truck_num][1];
        z = truckLocations[map_name][truck_num][2];
    } else {
        throw std::runtime_error("Invalid map name or truck index for location.");
    }

    return carla::geom::Location(x,y,z);
}

carla::geom::Rotation GetTruckRotation(int truck_num, std::string map_name) {
    float pitch = 0.0f , yaw = 0.0f ,roll =0.0f;
    if (truckRotations.find(map_name) != truckRotations.end() && truckRotations[map_name].find(truck_num) != truckRotations[map_name].end()) {
        pitch = truckRotations[map_name][truck_num][0];
        yaw = truckRotations[map_name][truck_num][1];
        roll = truckRotations[map_name][truck_num][2];
    } else {
        throw std::runtime_error("Invalid map name or truck index for rotation.");
    }
    return carla::geom::Rotation(pitch,yaw,roll);
}

void destroy_actor_by_role_name(const std::string &role_name) {
    auto actors = world->GetActors();
    for (auto actor : *actors) {
        if (actor == nullptr) {
            continue;
        }

        try {
            for (const auto &attribute : actor->GetAttributes()) {
                if (attribute.GetId() == "role_name" && attribute == role_name) {
                    std::cerr << "Destroying existing actor with role_name=" << role_name << '\n';
                    actor->Destroy();
                    break;
                }
            }
        } catch (const std::exception &) {
            continue;
        }
    }
}

void connect_to_carla(int truck_num) {
    //Connecting to CARLA server;
    client = new cc::Client(host, port);
    client->SetTimeout(40s);
    std::cerr << truck_num << " Truck connected to CARLA server" << '\n';
    world = new cc::World(client->GetWorld());
    blueprint_library = world->GetBlueprintLibrary();
}

void generate_truck(int truck_num, std::string map_name) {
    std::mt19937_64 rng((std::random_device())());  
    const std::string trailer_role_name = "trailer" + std::to_string(truck_num);
    const std::string truck_role_name = "truck" + std::to_string(truck_num);

    destroy_actor_by_role_name(trailer_role_name);
    destroy_actor_by_role_name(truck_role_name);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Get a Truck blueprint.
    auto truck = blueprint_library->Filter("dafxf");
    auto blueprint_truck = RandomChoice(*truck, rng);
    blueprint_truck.SetAttribute("role_name", truck_role_name);

    auto trailer = blueprint_library->Filter("trailer");
    if (trailer->size() == 0u) {
        throw std::runtime_error("Trailer blueprint not found.");
    }
    auto blueprint_trailer = RandomChoice(*trailer, rng);
    blueprint_trailer.SetAttribute("role_name", trailer_role_name);

    carla::geom::Location TruckLocation = GetTruckLocation(truck_num,map_name);
    carla::geom::Rotation TruckRotation = GetTruckRotation(truck_num,map_name);
    carla::geom::Transform transform(TruckLocation,TruckRotation);

    auto carla_map = world->GetMap();
    if (carla_map != nullptr) {
        auto waypoint = carla_map->GetWaypoint(
            TruckLocation,
            true,
            static_cast<int32_t>(carla::road::Lane::LaneType::Driving));
        if (waypoint != nullptr) {
            transform = waypoint->GetTransform();
            transform.location.z += 0.3f;
        }
    }

    carla::geom::Transform trailer_transform = transform;
    trailer_transform.location -= trailer_transform.GetForwardVector() * 5.2f;

    auto actor_trailer = world->SpawnActor(blueprint_trailer, trailer_transform);
    std::cout << "Spawned " << actor_trailer->GetDisplayId() << '\n';
    auto vehicle_trailer = boost::static_pointer_cast<cc::Vehicle>(actor_trailer);

    auto actor_truck = world->SpawnActor(blueprint_truck, transform);
    std::cout << "Spawned " << actor_truck->GetDisplayId() << '\n';
    auto vehicle_truck = boost::static_pointer_cast<cc::Vehicle>(actor_truck);

    // Force both actors into a stationary state at spawn time so they do not
    // creep before the ROS control path explicitly starts the fleet.
    carla::rpc::VehicleControl parked_control;
    parked_control.throttle = 0.0f;
    parked_control.brake = 1.0f;
    parked_control.steer = 0.0f;
    parked_control.reverse = false;
    parked_control.manual_gear_shift = false;
    parked_control.hand_brake = true;

    vehicle_truck->SetTargetVelocity(cg::Vector3D(0.0f, 0.0f, 0.0f));
    vehicle_truck->SetTargetAngularVelocity(cg::Vector3D(0.0f, 0.0f, 0.0f));
    vehicle_truck->ApplyControl(parked_control);

    vehicle_trailer->SetTargetVelocity(cg::Vector3D(0.0f, 0.0f, 0.0f));
    vehicle_trailer->SetTargetAngularVelocity(cg::Vector3D(0.0f, 0.0f, 0.0f));


    rclcpp::executors::MultiThreadedExecutor executor; 
    auto node_camera = std::make_shared<FrontCameraPublisher>(actor_truck);
    //auto node_radar = std::make_shared<FrontRadarPublisher>(actor_truck);
    auto node_lidar = std::make_shared<FrontLidarPublisher>(actor_truck);
    auto node_control = std::make_shared<TruckControl>(vehicle_truck);
    auto node_status = std::make_shared<TruckStatusPublisher>(vehicle_truck);

    executor.add_node(node_camera);
    //executor.add_node(node_radar);
    executor.add_node(node_lidar);
    executor.add_node(node_control);
    executor.add_node(node_status);

    executor.spin(); 



    vehicle_trailer->Destroy();
    vehicle_truck->Destroy();
}


int main(int argc, char *argv[]) {
    try {
        rclcpp::init(argc, argv);

        int truck_num = 1; // Default value
        std::string map_name = "IHP"; //Default map

        std::string prefix_truck_id("--truck_id=");
        std::string prefix_map_name("--map=");
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.find(prefix_truck_id) == 0) {
                truck_num = std::atoi(arg.substr(prefix_truck_id.length()).c_str());
            }
            else if (arg.find(prefix_map_name) == 0) {
                map_name = arg.substr(prefix_map_name.length());
            }
        }

        if(truck_num == 0 ) {
            std::cout << "Truck Number : " << truck_num << std::endl;
            std::cout << "Map Name : " << map_name << std::endl;            
        }

        connect_to_carla(truck_num);
        generate_truck(truck_num,map_name);

        rclcpp::shutdown();
    } 
    catch (const std::exception& e) {
        std::cerr << "Unhandled Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
