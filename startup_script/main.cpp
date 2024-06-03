#include <iostream>
#include <string>
#include <cstdio>
#include <memory>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

std::string exec(const char* cmd) {
  char buffer[128];
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer,sizeof(buffer), pipe.get()) != nullptr) {
    result += buffer;
  }
  return result;
}

static int connect_to_conf_server()
{
	struct sockaddr_in confServerAddr;
	confServerAddr.sin_family = AF_INET;
	confServerAddr.sin_port = htons(32121);
	confServerAddr.sin_addr.s_addr = inet_addr("10.42.0.1");

	int cnf_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (cnf_socket  == -1) {
		throw std::runtime_error("message socket creation failed");
	}
	for (;;) {
		try {
			if (connect(cnf_socket , (struct sockaddr*)&confServerAddr, sizeof(confServerAddr)) == -1) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			} else break;
		} catch (const std::exception& e) {
			close(cnf_socket);
			throw std::runtime_error(e.what());
		}
	}
	return cnf_socket;
}

int main() {
  try {
    std::string camera_repository = "/home/armolina/projects/arducam-apps";
    std::string camera_binaries = "/build/apps/arducam-raw";
    std::string command_options = " -t 0 -o tcp://10.42.0.1:32233 --message-ip tcp://10.42.0.1:32211 --shutter 1ms  --gain 1 --awbgains 1,1 --nopreview ";
    std::string command = "ifconfig | grep inet ";
    // Retry for 5 minutes until the Jetson provide IP to the Raspberry
    for (int i = 0; i <= 300; ++i) {
      std::string output = exec(command.c_str());
      auto ip_position = output.find("inet ");
      if ( ip_position != std::string::npos) {
        auto ip = output.substr(ip_position + 5, output.find(' ', ip_position + 5) - ip_position + 5);
        // This is the default IP that linux seems to assign to a computer
        // connected whit another directly
        if (ip.find("10.42.") != std::string::npos) {
          std::cout << "Found IP: " << ip << std::endl;
		  auto cnf_socket = connect_to_conf_server();
		  auto resolution_options = std::string();
		  while (true) {
			  char buffer[1024];
			  int bytesReceived = recv(cnf_socket, buffer, sizeof(buffer), 0);
			  // This condition prevents the code to put in a queue empty messages sent by the server.
			  if (bytesReceived >= 3) {
				  buffer[bytesReceived] = '\0';
				  resolution_options = std::string(buffer);
				  break;
			  } else {
				  std::cerr << "Connection closed by server or error occurred\n";
				  break;
			  }
		  }

		  auto camera_command = camera_repository + camera_binaries + command_options + resolution_options;
		  std::cout << " =========================\n " << camera_command << " \n=========================\n ";
		  int result = system(camera_command.c_str());
		  return result;
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
      } 
    }
	std::cout << "Jetson not found after retrying for 30 seconds.\n";
	return 1;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}
