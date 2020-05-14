#!/usr/bin/env python  
#-*- coding: utf-8 -*-

#!/usr/bin/env python
# coding: utf-8
import socket, time
serverIp = '219.224.167.239'
tcpPort = 9998
msg = "test"
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((serverIp, tcpPort))
time.sleep(1)
s.send(msg)
s.close()