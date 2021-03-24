/* demo_client.c - code for example client program that uses TCP 
*
* Code based off Brian Hutchinson's demo_client.c
*
* Bo Sullivan
* prog3_participant.c
* Winter 2021 CSC 367
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
#include <stdbool.h>
#include <ctype.h>
#include "prog3.h"


char* msgToSend;

/* checkAll takes a char* guess (actually name)
*  Returns a bool. True indicating name checked is valid.
*  Intended use: Making sure the username to send is between:
* 'a'-'z' || 'A'-'Z' || '0'-'9' || '_'
*/
bool checkAll(char* guess) {
        for (int i = 0; i < strlen(guess); i++) {

        if ((guess[i] < 48 || guess[i] > 57) && (guess[i] != 95)
        && (tolower(guess[i]) < 97 || tolower(guess[i]) > 122)){
            return false;
        }
    }
    return true;
    
}
/* get_username takes an int sd
*  Returns a bool. True indicates name was accepted by server.
*  Intended use is getting the username input and sending it to the server
*  and then validating the input. Used to loop until name is valid or timeout
*/
bool get_username(int sd) {
	char user_validate;
	int valid = 0;
	int n;
	//scanf not a fan of char* with  assigning value.
	char userName[10];
	//char* userName;
	while (valid == 0) {

		printf("Please enter a username 1-10 characters long: ");
		scanf("%10sd", userName);
		

		if (strlen(userName) == 0) {
			printf("Invalid from size 0 or space\nPlease enter new name:");
			valid = 0;
			continue;
		}

		bool b = checkAll(userName);
		if (b) { valid = 1;}
	}

	uint8_t nameLength = strlen(userName);

	//send size
	send(sd, &nameLength, sizeof(nameLength), MSG_NOSIGNAL);

	//send name
	send(sd, userName, nameLength, MSG_NOSIGNAL);

	n = recv(sd, &user_validate, sizeof(user_validate), 0);
	if ( n == 0) exit(EXIT_FAILURE);
	
	if (user_validate == 'Y') {
		return true;
	}
	else if (user_validate == 'T') {
		printf("username taken.\n");
		return false;
	}
	else if (user_validate == 'I') {
		printf("username invalid.\n");
		return false;
	}
	else {
		return false;
	}
}
/* chat takes an int sd
*  Returns an int, loops endlessly while active connection
*  Intending use: If usename is valid and accepted by server, continually 
*  prompt participant for message to send. "@name " would signify a priv msg
*/
int chat(int sd) {

    while (1) {
		char *usermsg;
		size_t len = 0;
		printf("Enter message: ");
		uint16_t msg_len =  getline(&usermsg, &len, stdin)-1;

		if (msg_len < 1) {
		//continue/exit? prompt for more input or close connection?
			continue;
		}

		if (msg_len > 1024) {
		usermsg[1024] = '\0';
		//maybe usrmsg[1024] = '0'
		msg_len = 1024;
		}
		uint16_t true_len;
		true_len = htons(msg_len);

		//send size -- host to net long or short?
		int s = send(sd, &true_len, sizeof(true_len), MSG_NOSIGNAL);
		if (s == 0) return 0;
		//send msg
		s = send(sd, usermsg, msg_len, MSG_NOSIGNAL);
		free(usermsg);
		continue;
    }
    return 0;
}

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

	/* Connect the socket to the specified server. */
	if (connect(sd, (struct sockaddr *)&sad, sizeof(sad)) < 0) {
		fprintf(stderr,"connect failed\n");
		exit(EXIT_FAILURE);
	}

	/* Repeatedly read data from socket and write to user's screen. */

	char can_connect;

	n = recv(sd, &can_connect, sizeof(can_connect), 0);
	if ( n == 0) exit(EXIT_FAILURE);
	
	if (can_connect == 'Y') {
		bool b = get_username(sd);

		while (!b) {
			get_username(sd);
		}
		chat(sd);
		close(sd);
		exit(0);
	}
	else  {
		printf("Server is currently Full.\n");
		close(sd);
		exit(0);
	}
}

