import socket,dill
import time

class clnt_sckt:
    def __init__(self, hostaddr, targetaddr) -> None:
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.bind(hostaddr)
        self.ad = targetaddr
        self.testing()

    def snd(self,**kwargs) -> None:
        try:
            para = []
            for key, value in kwargs.items():
                para.append((key, value))
            self.client_socket.sendto(dill.dumps(para), self.ad)
        except OSError as e:
            print(f"Sending {para} failed. {e}")

    def sndstr(self, s) -> None:
        self.client_socket.sendto(s.encode(), self.ad)
        return

    def terminate_sckt(self) -> None:
        self.client_socket.close()
    
    def testing(self) -> None:
        print("Waiting for connection...", flush=True)
        while True:
            self.client_socket.sendto("Handshake".encode('utf-8'), self.ad)
            data, address = self.client_socket.recvfrom(1024)
            if data.decode() == "Confirmative":
                print("Connection established.", flush=True)
                time.sleep(0.4)
                return
    
    def listening(self) -> str:
        while True:
            data, addr = self.client_socket.recvfrom(1024)
            print(data.decode())
            
