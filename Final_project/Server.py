from functions import *
import select


args = args_parser()
HEADER_LENGTH = 10
PORT = args.port
SERVER = socket.gethostbyname(socket.gethostname())
THRESHOLD = args.bond

ADDR = (SERVER, PORT)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  #set reuse the address

server.bind(ADDR)
server.listen()
sockets_list = [server]
clients = []
Weights = []

print('Listening for connections on ', SERVER)

com_time = 0
while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    for notified_socket in read_sockets:
        if notified_socket == server:
            client_socket, client_address = server.accept()

            sockets_list.append(client_socket)
            clients.append(client_socket)
            print('Accepted new connection from ', client_address)

            msg_recv = recv_msg(client_socket)
            if msg_recv is False:
                continue
            Weights.append(msg_recv[1])

            if len(Weights) == THRESHOLD:
                new_weights = FedAvg(Weights)
                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', new_weights])
                Weights.clear()
                print('[NEW FINISHED]', '\n')
                com_time += 1
        else:
            msg_recv = recv_msg(notified_socket)
            if msg_recv is False:
                print('[FALSE] Closed connection from: ', notified_socket, '...')
                sockets_list.remove(notified_socket)
                clients.remove(notified_socket)
                continue
            Weights.append(msg_recv[1])
            if len(Weights) == THRESHOLD:
                new_weights = FedAvg(Weights)
                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', new_weights])
                Weights.clear()
                print(com_time, '[REPEAT FINISHED]', '\n')
                com_time += 1

    for notified_socket in exception_sockets:
        # Remove from list for socket.socket()
        sockets_list.remove(notified_socket)
        clients.remove(notified_socket)
