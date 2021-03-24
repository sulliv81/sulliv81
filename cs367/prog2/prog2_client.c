/* prog2_client.c - code for example client program that uses TCP 
 *
 * This code is based on demo_client.c by Brian Hutchinson
 *
 * Project 2 prog2_client.c
 * February 20, 2021
 * CSCI 367 Brian Hutchinson Winter 2021
 * Dylan Thompson & Bo Sullivan
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

/* macros for common code to catch send/recv errors inside a while loop */
#define recv_or_die(A,B,C,D) {\
n = recv(A, B, C, D);\
if (n == 0) { socket_open = false; break; }\
}
#define send_or_die(A,B,C,D) {\
n = send(A, B, C, D);\
if (n < 0) { socket_open = false; break; }\
}

int main( int argc, char **argv) {
	struct hostent *ptrh; /* pointer to a host table entry */
	struct protoent *ptrp; /* pointer to a protocol table entry */
	struct sockaddr_in sad; /* structure to hold an IP address */
	int sd; /* socket descriptor */
	int port; /* protocol port number */
	char *host; /* pointer to host name */
	int n; /* number of characters read */

	uint8_t board_size; /* size of the game board */
	uint8_t timeout; /* turn timeout in seconds */

	char guess[256]; /* buffer to hold client guess - 255 max length + 1 null terminator*/
	char board[255]; /* buffer to hold current game board */
	
	memset((char *)&sad,0,sizeof(sad)); /* clear sockaddr structure */
	sad.sin_family = AF_INET; /* set family to Internet */

	/* Validate CLI args */
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
	
	/* Get identifier telling client which player it is */
	char which_player;
	n = recv(sd, &which_player, sizeof(which_player), 0);
	if (n == 0) exit(EXIT_FAILURE);

	if (which_player == '1') {
		printf("You are Player 1... the game will begin when Player 2 joins...\n");
	} else if (which_player == '2') {
		printf("You are Player 2...\n");
	} else {
		fprintf(stderr, "Error: server sent invalid value for which_player\n");
		exit(EXIT_FAILURE);
	}
		
	/* Get board size */
	n = recv(sd, &board_size, sizeof(board_size), 0);
	if (n == 0) exit(EXIT_FAILURE);
	printf("Board size: %d\n", board_size);

	/* Get turn timeout */
	n = recv(sd, &timeout, sizeof(timeout), 0);
	if (n == 0) exit(EXIT_FAILURE);
	printf("Seconds per turn: %d\n", timeout);

	bool socket_open = true;

	/* Round loop */
	while (socket_open) { // and while neither player has won
		/* Get current score */
		uint8_t p1_score, p2_score;
		recv_or_die(sd, &p1_score, sizeof(p1_score), 0);
		recv_or_die(sd, &p2_score, sizeof(p2_score), 0);

		uint8_t my_score = which_player == '1' ? p1_score : p2_score;
		uint8_t their_score = which_player == '1' ? p2_score : p1_score;

		/* Check whether either player won */
		if (my_score >= 3) {
			printf("You won!\n\n");
			break;
		} else if (their_score >= 3) {
			printf("You lost!\n\n");
			break;
		}
		
		/* Get round number */
		uint8_t round_number;
		recv_or_die(sd, &round_number, sizeof(round_number), 0);
		printf("\nRound %d...\n", round_number);
		printf("Score is %d-%d\n", my_score, their_score);

		/* Get game bord for current round */
		recv_or_die(sd, board, board_size, MSG_WAITALL);
		printf("Board: ");
		for (int i = 0; i < board_size; i++) {
			printf("%c ", board[i]);
		}
		printf("\n");

		char whose_turn;
		/* Turn loop */
		while (socket_open) {

			/* Get identifier telling client whether it's their turn */
			recv_or_die(sd, &whose_turn, sizeof(whose_turn), 0);

			if (whose_turn == 'Y') {

				/* Get guess from CLI user */
				printf("Your turn, enter word: ");
				scanf("%255s", guess);

				/* Send guess */
				uint8_t guess_len = strlen(guess);
				send_or_die(sd, &guess_len, sizeof(guess_len), MSG_NOSIGNAL);
				send_or_die(sd, guess, guess_len, MSG_NOSIGNAL);

				/* Get whether guess was valid */
				uint8_t valid;
				recv_or_die(sd, &valid, sizeof(valid), 0);

				if (valid) {
					printf("Valid word!\n");
					// continue round
				} else {
					printf("Invalid word!\n");
					break; // end_round
				}
			} else if (whose_turn == 'N') {
				printf("Please wait for opponent to enter word...\n");

				/* Get guess length from server, or 0 if guess was invalid */
				uint8_t guess_len;
				recv_or_die(sd, &guess_len, sizeof(guess_len), 0);

				if (guess_len == 0) {
					printf("Opponent lost the round!\n");
					break; // end round
				} else {
					/* Get the actual guess string */
					recv_or_die(sd, guess, guess_len, MSG_WAITALL);
					guess[guess_len] = '\0'; // add null terminator
					printf("Opponent entered \"%s\"\n", guess);
					// continue round
				}
			} else {
				fprintf(stderr, "Error: server sent invalid value for whose_turn\n");
				exit(EXIT_FAILURE);
			}
		}
	}
	
	close(sd);
	exit(EXIT_SUCCESS);
}

