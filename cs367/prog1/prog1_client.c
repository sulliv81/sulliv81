/* demo_client.c - code for example client program that uses TCP 
* Very largely bsaed off demo code provided in useful links from canvas.
* Sends first guess to server and server maintaints a secret number and
* responds appropriately until client meets win condition of guessing
* the secret number on the first try. Guess auxilliary method to get the
* int16_t guess from user and send that.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

int roundc;
int guessc;
int guess;
int16_t gess;
int16_t getGess();
/*------------------------------------------------------------------------
* Program: demo_client
*
* Purpose: allocate a socket, connect to a server, and print all output
*
* Syntax: ./demo_client server_address server_port
*
* server_address - name of a computer on which server is executing
* server_port    - protocol port number server is using
*
*------------------------------------------------------------------------
*/
int main( int argc, char **argv) {
	struct hostent *ptrh; /* pointer to a host table entry */
	struct protoent *ptrp; /* pointer to a protocol table entry */
	struct sockaddr_in sad; /* structure to hold an IP address */
	int sd; /* socket descriptor */
	int port; /* protocol port number */
	char *host; /* pointer to host name */
	int n; /* number of characters read */
	char buf[1000]; /* buffer for data from the server */
	int z;
	int gameActive;
	//guess = getGuess();
	//gess = getGess();
	memset((char *)&sad,0,sizeof(sad)); /* clear sockaddr structure */
	sad.sin_family = AF_INET; /* set family to Internet */

	if( argc != 3 ) {
		fprintf(stderr,"Error: Wrong number of arguments\n");
		fprintf(stderr,"usage:\n");
		fprintf(stderr,"./client server_address server_port\n");
		exit(EXIT_FAILURE);
	}

	port = atoi(argv[2]); /* convert to binary */
	if (port > 0) /* test for legal value */
		sad.sin_port = htons((u_short)port);
	else {
		fprintf(stderr,"Error: bad port number %s\n",argv[2]);
		exit(EXIT_FAILURE);
	}

	host = argv[1]; /* if host argument specified */

	/* Convert host name to equivalent IP address and copy to sad. */
	ptrh = gethostbyname(host);
	if ( ptrh == NULL ) {
		fprintf(stderr,"Error: Invalid host: %s\n", host);
		exit(EXIT_FAILURE);
	}

	memcpy(&sad.sin_addr, ptrh->h_addr, ptrh->h_length);

	/* Map TCP transport protocol name to protocol number. */
	if ( ((long int)(ptrp = getprotobyname("tcp"))) == 0) {
		fprintf(stderr, "Error: Cannot map \"tcp\" to protocol number");
		exit(EXIT_FAILURE);
	}

	/* Create a socket. */
	sd = socket(PF_INET, SOCK_STREAM, ptrp->p_proto);
	if (sd < 0) {
		fprintf(stderr, "Error: Socket creation failed\n");
		exit(EXIT_FAILURE);
	}

	//while this value (set to int) is not 0, keep connection open
	/* Connect the socket to the specified server. */
	//int xxt; // setting connection, while above 0
	//xxt = connect(sd, (struct sockaddr *)&sad, sizeof(sad)));

	if (connect(sd, (struct sockaddr *)&sad, sizeof(sad)) < 0) {
		fprintf(stderr,"connect failed\n");
		exit(EXIT_FAILURE);
	}
	gameActive = 0;
	gess = getGess();
	z = send(sd, &gess, sizeof(gess), 0);//sends first guess to server
	n = recv(sd, buf, sizeof(buf), 0);
	printf("n outside while is: %d\n", n);
	//while still receiving (game active) from server
	while (n > 0 && gameActive != 1) 
	{
		if (buf[0] == 'w') { 
			roundc++;
			guessc = 0;  
			printf("Correct, advance to next round.\n");
			gess = getGess();
			z = send(sd, &gess, sizeof(gess), 0);
			
		}
		else if (buf[0] == 'W') {
			printf("You win!\n");
			roundc = 0;
			guessc = 0;
			gameActive = 1;
			close(sd);
			exit(EXIT_SUCCESS);
		} 
		else if(buf[0] == 'l') {
			guessc++;
			printf("too low, guess again.\n");
			gess = getGess();
			z = send(sd, &gess, sizeof(gess), 0);
		}
		else if(buf[0] == 'L') {
			guessc =  0;
			printf("too low, and pushed hidden number out of bounds. You lose...\n");
			gameActive = 1;
		}
		else if(buf[0] == 'H') {
			guessc = 0;
			printf("too high, and pushed hidden number out of bounds. You lose...\n");
			gameActive = 1;
		}
		else if(buf[0] == 'h') {
			guessc++;
			printf("too high, guess again.\n");
			gess = getGess();
			z = send(sd, &gess, sizeof(gess), 0);
		}
		n = recv(sd, buf, sizeof(buf), 0);
	} 
	close(sd); 
	exit(EXIT_SUCCESS);
}
int16_t getGess() {
	int guess;
	printf("Round %d, turn %d\nEnter guess:", roundc, guessc);
	scanf("%d", &guess);
	if (guess < -255 || guess > 255) {
		printf("guess outside bounds, resulting in a guess of 0.\n");
		guess = 0;
	}
	return guess;
}
