#include "TruckControl.hpp"

TruckControl::TruckControl(boost::shared_ptr<carla::client::Vehicle> vehicle_)
    : Node("truck_control_node", rclcpp::NodeOptions()
               .allow_undeclared_parameters(true)
           .automatically_declare_parameters_from_overrides(true)),Vehicle_(vehicle_) {
    
    this->get_parameter_or("steer_topic_name",steer_topic_name,std::string("steer"));
    this->get_parameter_or("throttle_topic_name",throttle_topic_name,std::string("throttle"));
    this->get_parameter_or("initial_hand_brake",initial_hand_brake,true);
    this->get_parameter_or("steer_input_max_deg", steer_input_max_deg, 30.0f);


    SteerSubscriber_ = this->create_subscription<std_msgs::msg::Float32>(steer_topic_name, 1, std::bind(&TruckControl::SteerSubCallback, this, std::placeholders::_1));
    ThrottleSubscriber_ = this->create_subscription<std_msgs::msg::Float32>(throttle_topic_name, 1, std::bind(&TruckControl::ThrottleSubCallback, this, std::placeholders::_1));

    this->control = carla::rpc::VehicleControl();
    this->control.throttle = 0.0f;
    this->control.brake = 1.0f;
    this->control.steer = 0.0f;
    this->control.reverse = false;
    this->control.manual_gear_shift = false;
    this->control.hand_brake = initial_hand_brake;
    Vehicle_->ApplyControl(control);
}


void TruckControl::SteerSubCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    float max_deg = steer_input_max_deg > 1.0f ? steer_input_max_deg : 30.0f;
    float control_value = msg->data / max_deg;
    if (control_value > 1.0) this->control.steer = 1.0f;
    else if (control_value < -1.0) this->control.steer = -1.0f;
    else this->control.steer = control_value;
    Vehicle_->ApplyControl(control);
}

void TruckControl::ThrottleSubCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    float control_value = msg->data;
    if (this->control.hand_brake == true ) {
        if (control_value > 0.01f) {
            this->control.hand_brake = false;
            this->control.brake = 0.0f;
            this->control.throttle = control_value;
        }
        else {
            this->control.throttle = 0.0f;
            this->control.brake = 1.0f;
        }
        Vehicle_->ApplyControl(control);
    }
    else {
        this->control.reverse = false;
        this->control.manual_gear_shift = false;
        if (control_value >= 0) {
            this->control.throttle = control_value;
            this->control.brake = 0;
        }
        else if (control_value < 0) {
            this->control.throttle = 0;
            this->control.brake = -control_value;
        }
        Vehicle_->ApplyControl(control);
    }
}
