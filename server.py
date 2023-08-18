import socket, time, proc, dill, threading
from Sources import chart, client
from Sources.config import server_ip, client_ip, monitor_ip, main_port

class srvr_sckt:
    def __init__(self, addr) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(addr)
        try:
            if addr == (server_ip, main_port):
                self.testing()
        except:
            print("Jump connecting process")
            pass

    def testing(self) -> None:
        print("Waiting for connection...", flush = True)
        while False:
            self.server_socket.sendto(b'HHH', (client_ip, main_port))
            time.sleep(0.5)
        while True:
            data, addr = self.server_socket.recvfrom(1024)
            print(f"receive from: {addr}")
            if data.decode() == "Handshake":
                print("Connection Established", flush = True)
                self.server_socket.sendto(b'Confirmative', addr)
                return

    def run(self) -> list or str:
        try:
            data, addr = self.server_socket.recvfrom(10000)
            return dill.loads(data)
        except:
            return []
        
    def sending(self, s, adr) -> None:
        self.server_socket.sendto(s.encode(), adr)
        return
    
    def receiving(self) -> str:
        data, addr = self.server_socket.recvfrom(10000)
        return data.decode()

    def terminate(self) -> None:
        self.server_socket.close()

if __name__ == "__main__":
    srvr = srvr_sckt((server_ip, main_port))

    dotList = []
    pic_fa = chart.cht(9)
    
    def refresh():
        pkg_count = 0
        while True:
            s = srvr.run()
            for key, value in s:
                if key == 'create':
                    srl = value
                    continue
                elif key == 'device':
                    srl = value
                    out = dotList[srl].run(s)
                    out += f" {pkg_count}"
                    pkg_count += 1
                    print(f'\r{out}', end = '', flush=True)
                    
                    try:
                        pic_fa.run(dotList[srl].fax_raw, 0)
                        pic_fa.run(dotList[srl].fay_raw, 1)
                        pic_fa.run(dotList[srl].faz_raw, 2)
                        pic_fa.run(dotList[srl].cali_fax, 3)
                        pic_fa.run(dotList[srl].cali_fay, 4)
                        pic_fa.run(dotList[srl].cali_faz, 5)
                        pic_fa.run(dotList[srl].cali_vel['x'], 6)
                        pic_fa.run(dotList[srl].cali_vel['y'], 7)
                        pic_fa.run(dotList[srl].cali_vel['z'], 8)
                        srvr.sending(f"{dotList[srl].fax} {dotList[srl].fay} {dotList[srl].faz}", (monitor_ip, 12570))
                        srvr.sending(f"{dotList[srl].cali_coor['x']} {dotList[srl].cali_coor['y']} {dotList[srl].cali_coor['z']}", (monitor_ip, 12571))
                        srvr.sending(f"{dotList[srl].ox} {dotList[srl].oy} {dotList[srl].oz}", (monitor_ip, 12572))
                        srvr.sending(f"{dotList[srl].cali_vel['x']} {dotList[srl].cali_vel['y']} {dotList[srl].cali_vel['z']}", (monitor_ip, 12573))
                        srvr.sending(f"{dotList[srl].cali_fax} {dotList[srl].cali_fay} {dotList[srl].cali_faz}", (monitor_ip, 12574))
                        srvr.sending(f"{dotList[srl].coor['x']} {dotList[srl].coor['y']} {dotList[srl].coor['z']}", (monitor_ip, 12575))
                    except Exception as e:
                        print(f"\rChart error: {e}")
                        pass
                    break
                mode = value
                dotList.append(proc.Dot(srl, mode))
        
    try:
        thr = threading.Thread(target=refresh)
        thr.start()
        pic_fa.exe()
    except KeyboardInterrupt:
        srvr.terminate()
