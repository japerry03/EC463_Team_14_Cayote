#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <string>
#include <algorithm>
#include <csignal>
#include <chrono>
#include <cerrno>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <librealsense2/rs.hpp>
#include <zmq.hpp>
#include <fstream>
#include <sys/stat.h>

/*
<LAST UPDATED ---> November 20th, 2025>
Steps to make this function:

1. Run the following command:


sudo apt-get install librealsense2-dev libzmq3-dev

2. Now, you must compile it with the correct libraries mentioned as well, so do:


g++ camera_daemon.cpp -o <whatever you name it> -lrealsense2 -L/usr/local/lib -I/usr/local/include -lpthread -lz



Once this is run, this code should compile correctly :)
*/


// DISCLAIMER: I did get ChatGPT's help when writing this code (I'm too tired to read that much C++ documentation...). As such,
// this code may have some glaring weakness I overlooked, or an error that I did not notice. Please thoroughly check this code. 


using namespace std;

// Success Code: 0
// Fail Code: -1


// Set up camera and read data:
static const int DEFAULT_WID = 640;
static const int DEFAULT_LEN = 480;
static const int DEFAULT_FPS = 30; // I never know why people want >30 FPS in their games. The human eye generally can't percieve more than 30 FPS. 60 is far more than enough
static const int DEFAULT_PORT = 5398; // Just pick a random port that isn't 0-1023 (That port range is generally reserved for ports with system services or specific apps)
static const char* DEFAULT_UDS_PATH = "/var/run/camera_daemon.sock";				     
static const char* DEFAULT_TOKEN_FILE = "/etc/camera_daemon/token";
static const long int SSIZE_MAX = 9223372036854775807;



atomic<bool> running(true);

void handle_signal(int){
	running = false; //
}

int set_nonblocking(int fd){
	int flags = fcntl(fd, F_GETFL, 0);
	if (flags == -1){
		return -1; // If there's no flag, return fail code.
	}
	return fcntl(fd, F_SETFL, flags | O_NONBLOCK); 
}

string read_token_file(const string &path){
	ifstream ifs(path);
	if(!ifs){
		cerr << "WARNING: Token File " << path << " not found or unreadable. TCP Authorization disabled\n";
		return "";
	}

	string token;
	getline(ifs, token);
	token.erase(token.find_last_not_of(" \n\r\t") + 1);
	token.erase(0, token.find_first_not_of(" \n\r\t"));
	return token;
}



struct Client {
	int fd;
	sockaddr_storage addr;
	socklen_t addrlen;
	bool is_uds;
};

class ClientManager {
	public:
		void add_client(Client c){
			lock_guard<mutex> lk(mutex_);
			clients_.push_back(c);
		}

		
		
		void broadcast_frame(const uint8_t* data, size_t size){
			lock_guard<mutex> lk(mutex_);
			uint32_t len_be = htonl(static_cast<uint32_t>(size));
			vector<int> to_remove;
			for(size_t i =0; i < clients_.size(); ++i){
				int fd = clients_[i].fd;
				ssize_t w = write_all(fd, (const uint8_t*)&len_be, sizeof(len_be));
				
				
				if(w < 0){
					to_remove.push_back((int)i); 
					continue;
				}
				w = write_all(fd, data, size);
				
				
				if(w < 0){
					to_remove.push_back((int)i);
					continue;
				}

			}
			for (int idx = (int)to_remove.size()-1; idx >= 0; --idx){
				int i = to_remove[idx];
				close(clients_[i].fd);
				clients_.erase(clients_.begin()+i);
			}
		}

		void close_all(){
			lock_guard<mutex> lk(mutex_);
			for(auto &c: clients_) close(c.fd);
			clients_.clear();
		}


	private:
		vector<Client> clients_;
		mutex mutex_;
		
		
		
		static ssize_t write_all(int fd, const uint8_t* buf, size_t len) {
			size_t total = 0;



            if((len - total) < SSIZE_MAX){
                cout << "Possible Integer Overflow... Ending..." << endl;
                return -1; // Best that it doesn't overflow lmao 
            }

			while (total < len) {
				ssize_t n = send(fd, buf + total, len - total, 0);
					if (n > 0) { 
						total += n; continue; 
					}
					
					if (n == 0) {
						return (ssize_t)total;
					}
					
					if (errno == EAGAIN || errno == EWOULDBLOCK) {
						// backpressure or slow client — treat as error and drop client
						return -1;
					}
				return -1;
			}
			return (ssize_t)total;
    }
		

};



void tcp_accept_loop(int listen_fd, ClientManager &mgr, const std::string &token) {
    // listening socket assumed non-blocking
    while (running) {
        sockaddr_storage addr;
        socklen_t addrlen = sizeof(addr);
        int client_fd = accept(listen_fd, (sockaddr*)&addr, &addrlen);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else {
                if (running) perror("accept");
                break;
            }
        }
        // Set non-blocking
        set_nonblocking(client_fd);

        // Read token line (up to 512 bytes or until newline) with a small timeout loop
        std::string received;
        char buf[512];
        bool auth_ok = false;
        if (token.empty()) {
            // No token configured: allow
            auth_ok = true;
        } else {
            // wait for newline but bounded time
            int tries = 0;
            while (tries++ < 50) { // ~50 * 10ms = 500ms
                ssize_t r = recv(client_fd, buf, sizeof(buf)-1, 0);
                if (r > 0) {
                    buf[r] = '\0';
                    received += std::string(buf, (size_t)r);
                    if (received.find('\n') != std::string::npos) break;
                } else if (r == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                } else {
                    break;
                }
            }
            // Trim and check
            size_t pos = received.find('\n');
            if (pos != std::string::npos) {
                std::string tok = received.substr(0, pos);
                // trim
                tok.erase(tok.find_last_not_of("\r\n\t ")+1);
                tok.erase(0, tok.find_first_not_of("\r\n\t "));
                if (!tok.empty() && tok == token) auth_ok = true;
            }
        }

        if (!auth_ok) {
            std::cerr << "TCP client failed auth; closing\n";
            close(client_fd);
            continue;
        }

        Client c;
        c.fd = client_fd;
        c.addrlen = addrlen;
        c.addr = addr;
        c.is_uds = false;
        mgr.add_client(c);
        std::cerr << "TCP client accepted (fd=" << client_fd << ")\n";
    }
}



void uds_accept_loop(int uds_listen_fd, ClientManager &mgr) {
    while (running) {
        sockaddr_storage addr;
        socklen_t addrlen = sizeof(addr);
        int client_fd = accept(uds_listen_fd, (sockaddr*)&addr, &addrlen);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else {
                if (running) perror("uds accept");
                break;
            }
        }
        set_nonblocking(client_fd);
        Client c;
        c.fd = client_fd;
        c.addrlen = addrlen;
        c.addr = addr;
        c.is_uds = true;
        mgr.add_client(c);
        std::cerr << "UDS client accepted (fd=" << client_fd << ")\n";
    }
}


int create_tcp_listener(int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) { perror("socket"); return -1; }
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("bind");
        close(sock);
        return -1;
    }
    if (listen(sock, 16) == -1) { perror("listen"); close(sock); return -1; }
    set_nonblocking(sock);
    return sock;
}



int create_uds_listener(const std::string &path) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) { perror("socket uds"); return -1; }
    // unlink if existing
    unlink(path.c_str());
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path)-1);
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("bind uds");
        close(sock);
        return -1;
    }
    if (listen(sock, 8) == -1) { perror("listen uds"); close(sock); return -1; }
    set_nonblocking(sock);
    return sock;
}



int main(int argc, char** argv){
    int width = DEFAULT_WID;
    int length = DEFAULT_LEN;
    int fps = DEFAULT_FPS;
    int port = DEFAULT_PORT;
    std::string uds_path = DEFAULT_UDS_PATH;
    std::string token_file = DEFAULT_TOKEN_FILE;

    for (int i=1;i<argc;i++) {
        std::string s(argv[i]);
        if (s == "--width" && i+1<argc) width = atoi(argv[++i]);
        else if (s == "--length" && i+1<argc) length = atoi(argv[++i]);
        else if (s == "--fps" && i+1<argc) fps = atoi(argv[++i]);
        else if (s == "--port" && i+1<argc) port = atoi(argv[++i]);
        else if (s == "--uds" && i+1<argc) uds_path = argv[++i];
        else if (s == "--token-file" && i+1<argc) token_file = argv[++i];
        else if (s == "--help") {
            std::cout << "camera_daemon [--width W] [--length L] [--fps F] [--port P] [--uds PATH] [--token-file PATH]\n";
            return 0;
        }
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::string token = read_token_file(token_file);
    if (!token.empty()) {
        std::cerr << "TCP authentication token loaded (length=" << token.size() << ")\n";
    } else {
        std::cerr << "No token loaded; TCP connections will be allowed without token.\n";
    }

    int tcp_listen = create_tcp_listener(port);
    int uds_listen = create_uds_listener(uds_path);
    if (tcp_listen == -1 && uds_listen == -1) {
        std::cerr << "No listeners available. Exiting.\n";
        return 1;
    }

    if (uds_listen != -1) {
        // chmod to 0660
        chmod(uds_path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
        std::cerr << "UDS socket created: " << uds_path << " (set to 660)\n";
    }

    ClientManager mgr;
    std::thread tcp_thread;
    std::thread uds_thread;
    if (tcp_listen != -1) tcp_thread = std::thread(tcp_accept_loop, tcp_listen, std::ref(mgr), token);
    if (uds_listen != -1) uds_thread = std::thread(uds_accept_loop, uds_listen, std::ref(mgr));

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, width, length, RS2_FORMAT_BGR8, fps);
    try {
        pipe.start(cfg);
    } catch (const rs2::error &e) {
        std::cerr << "RealSense error: " << e.what() << "\n";
        running = false;
    }

    std::cerr << "Camera daemon running. Streaming " << width << "x" << length << " @ " << fps << "fps\n";

    while (running) {
        try {
            rs2::frameset frames = pipe.wait_for_frames(1000); // 1000ms timeout
            rs2::video_frame color = frames.get_color_frame();
            if (!color) continue;
            const uint8_t* data = reinterpret_cast<const uint8_t*>(color.get_data());
            size_t frame_size = color.get_height() * color.get_width() * color.get_bytes_per_pixel();
            mgr.broadcast_frame(data, frame_size);
        } catch (const rs2::error &e) {
            std::cerr << "Realsense runtime error: " << e.what() << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << "\n";
        }
    }

    std::cerr << "Shutting down...\n";
    try { 
        pipe.stop(); 
    } catch(...) {
        
    }
    if (tcp_listen != -1) close(tcp_listen);
    if (uds_listen != -1) close(uds_listen);
    mgr.close_all();
    if (!uds_path.empty()) unlink(uds_path.c_str());

    if (tcp_thread.joinable()) tcp_thread.join();
    if (uds_thread.joinable()) uds_thread.join();

    std::cerr << "Exited cleanly.\n";
    return 0;
}
