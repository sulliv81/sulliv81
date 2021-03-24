/* demo_server.c - code for example server program that uses TCP 
* Very largely bsaed off demo code provided in useful links from canvas.
* I also included standard bool, if this is a problem I would glady change it
* assuming 0s and 1s are easy to modify to represent booleans. I just enjoy the
* true/false flags for people's readability, especially server side. Server side will
* maintain secret number and update accordingly as well as handle multiple clients
* running on the same server running their own version of the game and maintaing 
* their own secret value.
*/


#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#define QLEN 6 /* size of request queue */
int visits = 0; /* counts client connections */

/*------------------------------------------------------------------------
* Program: demo_server
*
* Purpose: allocate a socket and then repeatedly execute the following:
* (1) wait for the next connection from a client
* (2) send a short message to the client
* (3) close the connection
* (4) go back to step (1)
*
* Syntax: ./demo_server port
*
* port - protocol port number to use
*
*------------------------------------------------------------------------
*/

int main(int argc, char **argv) {
	struct protoent *ptrp; /* pointer to a protocol table entry */
	struct sockaddr_in sad; /* structure to hold server's address */
	struct sockaddr_in cad; /* structure to hold client's address */
	int sd, sd2; /* socket descriptors */
	int port; /* protocol port number */
	int alen; /* length of address */
	int optval = 1; /* boolean value when we set socket option */
	char buf[1000]; /* buffer for string the server sends */
	int z;
	int16_t H;
	H = atoi(argv[2]);

	if( argc != 3 ) {
		fprintf(stderr,"Error: Wrong number of arguments\n");
		fprintf(stderr,"usage:\n");
		fprintf(stderr,"./server server_port\n");
		exit(EXIT_FAILURE);
	}

	memset((char *)&sad,0,sizeof(sad)); /* clear sockaddr structure */
	sad.sin_family = AF_INET; /* set family to Internet */
	sad.sin_addr.s_addr = INADDR_ANY; /* set the local IP address */

	port = atoi(argv[1]); /* convert argument to binary */
	if (port > 0) { /* test for illegal value */
		sad.sin_port = htons((u_short)port);
	} else { /* print error message and exit */
		fprintf(stderr,"Error: Bad port number %s\n",argv[1]);
		exit(EXIT_FAILURE);
	}

	/* Map TCP transport protocol name to protocol number */
	if ( ((long int)(ptrp = getprotobyname("tcp"))) == 0) {
		fprintf(stderr, "Error: Cannot map \"tcp\" to protocol number");
		exit(EXIT_FAILURE);
	}

	/* Create a socket */
	sd = socket(PF_INET, SOCK_STREAM, ptrp->p_proto);
	if (sd < 0) {
		fprintf(stderr, "Error: Socket creation failed\n");
		exit(EXIT_FAILURE);
	}

	/* Allow reuse of port - avoid "Bind failed" issues */
	if( setsockopt(sd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0 ) {
		fprintf(stderr, "Error Setting socket option failed\n");
		exit(EXIT_FAILURE);
	}

	/* Bind a local address to the socket */
	if (bind(sd, (struct sockaddr *)&sad, sizeof(sad)) < 0) {
		fprintf(stderr,"Error: Bind failed\n");
		exit(EXIT_FAILURE);
	}

	/* Specify size of request queue */
	if (listen(sd, QLEN) < 0) {
		fprintf(stderr,"Error: Listen failed\n");
		exit(EXIT_FAILURE);
	}
	int cnt = 0;
	/* Main server loop - accept and handle requests */
	while (1) {
		alen = sizeof(cad);
		if ( (sd2=accept(sd, (struct sockaddr *)&cad, (socklen_t*)&alen)) < 0) {
			fprintf(stderr, "Error: Accept failed\n");
			exit(EXIT_FAILURE);
		}
		//connection with client made here after if above
		//now fork
		int pid = fork();
		//child
		if (pid == 0) {
			visits++;
			bool gameGoing = true;
			int16_t x;
			while (gameGoing) { 
				z = recv(sd2, &x, sizeof(x), 0);

				if ( z > 0 && gameGoing) {
					cnt++;
					//win condition
					if(x == H && cnt == 1) {
						sprintf(buf, "W\n");
						gameGoing = false;
					}
					//correct guess and reset condition
					else if(x == H && cnt > 1) {
						H = (H + (H - x));
						cnt = 0;
						sprintf(buf, "w\n");
					}
					else if (x < H) {
						H = H + (H - x);
						if (H < -256) {
							sprintf(buf, "L\n"); 
							gameGoing = false;
						}
						else if (H > 255) {
							sprintf(buf, "H\n"); 
							gameGoing = false;
						}
						else {
							sprintf(buf, "l\n");
						}
					}
					else if (x > H) {	
						H = H + (H - x);
						if (H < -256) {
							sprintf(buf, "L\n");
							gameGoing = false;
						}
						else if (H > 255) {
							sprintf(buf, "H\n"); 
							gameGoing = false;
						}
						else {
							sprintf(buf, "h\n");
						}
					}
					send(sd2,buf,strlen(buf),0);
				}
			}
			close(sd2); //closing child
			exit(EXIT_SUCCESS);
		}
		// closing parent
		else {
			close(sd2);
		}
	}
}
