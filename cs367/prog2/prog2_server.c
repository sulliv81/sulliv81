/* prog_server.c - code for example client program that uses TCP 
 *
 * This code is based on demo_server.c by Brian Hutchinson
 *
 * Project 2 prog2_server.c
 * February 20, 2021
 * CSCI 367 Brian Hutchinson Winter 2021
 * Dylan Thompson & Bo Sullivan
 */

#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include "trie.h"
#include <errno.h>
#include <ctype.h>
#define QLEN 6 /* size of request queue */

/* macros for common code to catch send/recv errors inside a while loop */
#define recv_or_die(A,B,C,D) {\
n = recv(A, B, C, D);\
if (n == 0) { socket_open = false; break; }\
}
#define send_or_die(A,B,C,D) {\
n = send(A, B, C, D);\
if (n < 0) { socket_open = false; break; }\
}

/* function declarations */
void generate_board(char* board, int board_size);
bool check_guess(char* guess, struct Trie* dict_trie, char* board, int size, struct Trie* guess_trie);

int main(int argc, char **argv) {

	struct protoent *ptrp; /* pointer to a protocol table entry */
	struct sockaddr_in sad; /* structure to hold server's address */
	struct sockaddr_in cad; /* structure to hold client's address */
	int sd, p1_sd, p2_sd; /* socket descriptors */
	uint alen; /* length of address */
	int optval = 1; /* boolean value when we set socket option */

	/* CLI args */
	int port; /* protocol port number */
	uint8_t timeout; /* turn timeout in seconds */
	uint8_t board_size; /* size of game board */
	char *dict_file; /* path to word list */
	
	/* Game data */
	uint8_t round = 1; /* curren t round number */
	uint8_t p1_score = 0; /* player 1's score */
	uint8_t p2_score = 0; /* player 2's score */
	char board[255]; /* buffer for game board */
	char guess[256]; /* buffer for player guess - 255 max len + 1 null terminator */
	int n; /* temp var to hold return vals from recv/send */

	if( argc != 5) {
		fprintf(stderr,"Error: Wrong number of arguments\n");
		fprintf(stderr,"usage:\n");
		fprintf(stderr,"./server server_port board_size timeout dict_file\n");
		exit(EXIT_FAILURE);
	}

	board_size = atoi(argv[2]);
	if (board_size == 0) {
		fprintf(stderr,"Error: board_size must be positive\n");
		exit(EXIT_FAILURE);
	}

	timeout = atoi(argv[3]);
	if (timeout == 0) {
		fprintf(stderr,"Error: timeout must be positive\n");
		exit(EXIT_FAILURE);
	}

	dict_file = argv[4];

	memset((char *)&sad,0,sizeof(sad)); /* clear sockaddr structure */
	sad.sin_family = AF_INET; /* set family to Internet */
	sad.sin_addr.s_addr = INADDR_ANY; /* set the local IP address */

	port = atoi(argv[1]); /* convert argument to binary */
	printf("Port was: %d\n", port);
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

	struct Trie* dict_trie = get_new_trie_node();
	struct Trie* guess_trie = get_new_trie_node();

	FILE *fp = fopen(dict_file, "r");
    if (fp == NULL) {
		fprintf(stderr, "Error: couldn't open dictionary wordlist\n");
        exit(EXIT_FAILURE);
	}

	/* Read word list into dict trie */
	char *line = NULL;
	size_t len = 0;
	int read;
    while ((read = getline(&line, &len, fp)) != -1) {
		insert_key(dict_trie, line);
    }
    fclose(fp);

	/* Main server loop - accept and handle requests */
	while (1) {

		/* Listen for player 1*/
		alen = sizeof(cad);
		if ( (p1_sd=accept(sd, (struct sockaddr *)&cad, &alen)) < 0) {
			fprintf(stderr, "Error: Accept failed\n");
			exit(EXIT_FAILURE);
		}

		/* Send game settings to player 1 */
		char which_player = '1';
		n = send(p1_sd, &which_player, sizeof(which_player), MSG_NOSIGNAL);
		if (n < 0) continue;
		n = send(p1_sd, &board_size, sizeof(board_size), MSG_NOSIGNAL);
		if (n < 0) continue;
		n = send(p1_sd, &timeout, sizeof(timeout), MSG_NOSIGNAL);
		if (n < 0) continue;

		/* Listen for player 2 */
		alen = sizeof(cad);
		if ( (p2_sd=accept(sd, (struct sockaddr *)&cad, &alen)) < 0) {
			fprintf(stderr, "Error: Accept failed\n");
			exit(EXIT_FAILURE);
		}

		/* Send game settings to player 2 */
		which_player = '2';
		n = send(p2_sd, &which_player, sizeof(which_player), MSG_NOSIGNAL);
		if (n < 0) continue;
		n = send(p2_sd, &board_size, sizeof(board_size), MSG_NOSIGNAL);
		if (n < 0) continue;
		n = send(p2_sd, &timeout, sizeof(timeout), MSG_NOSIGNAL);
		if (n < 0) continue;

		int pid = fork();
		if (pid == 0) { // child
			/* Close bound listening socket in child process */
			close(sd);

			bool socket_open = true;

			/* Round loop */
			while ((p1_score <= 3 && p2_score <= 3) && socket_open) {
				make_empty_trie(&guess_trie);

				/* Send scores */
				send_or_die(p1_sd, &p1_score, sizeof(p1_score), MSG_NOSIGNAL);
				send_or_die(p2_sd, &p1_score, sizeof(p1_score), MSG_NOSIGNAL);
				send_or_die(p1_sd, &p2_score, sizeof(p2_score), MSG_NOSIGNAL);
				send_or_die(p2_sd, &p2_score, sizeof(p2_score), MSG_NOSIGNAL);

				/* If either player won, end the game */
				if (p1_score == 3 || p2_score == 3) {
					break;
				}

				/* Send round number */
				send_or_die(p1_sd, &round, sizeof(round), MSG_NOSIGNAL);
				send_or_die(p2_sd, &round, sizeof(round), MSG_NOSIGNAL);

				/* Generate & send game board */
				generate_board(board, board_size);
				send_or_die(p1_sd, board, board_size, MSG_NOSIGNAL);
				send_or_die(p2_sd, board, board_size, MSG_NOSIGNAL);

				/* Figure out which player gets the first turn  */
				bool p1_turn = (round % 2 == 1); // player 1 starts on odd number rounds

				/* Turn loop */
				while (socket_open) {
					int active_sd = p1_turn ? p1_sd : p2_sd;
					int inactive_sd = p1_turn ? p2_sd : p1_sd;
					uint8_t *inactive_score = p1_turn ? &p2_score : &p1_score;

					/* Send identifier teling each player whether its their turn */
					char turn_val = 'Y';
					send_or_die(active_sd, &turn_val, sizeof(turn_val), MSG_NOSIGNAL);
					turn_val = 'N';
					send_or_die(inactive_sd, &turn_val, sizeof(turn_val), MSG_NOSIGNAL);

					/* Set timeout for player guess */
					struct timeval tv;
					tv.tv_sec = timeout;
					tv.tv_usec = 0;
					setsockopt(active_sd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(struct timeval));

					/* Get guess length */
					uint8_t guess_len;
					recv_or_die(active_sd, &guess_len, sizeof(guess_len), 0);

					uint8_t guess_valid;

					/* On timeout, end round */
					if ( n == -1 && errno == EAGAIN) {
						/* Send invalid guess signal to both players */
						guess_valid = 0;
						send_or_die(active_sd, &guess_valid, sizeof(guess_valid), MSG_NOSIGNAL);
						send_or_die(inactive_sd, &guess_valid, sizeof(guess_valid), MSG_NOSIGNAL);

						/* Increment score for inactive player */
						(*inactive_score)++;

						break; // end round
					}
						
					/* Get guess */ 
					recv_or_die(active_sd, guess, guess_len, MSG_WAITALL);
					guess[guess_len] = '\0'; // add null terminator
					
					/* Check the guess */
					guess_valid = check_guess(guess, dict_trie, board, board_size, guess_trie);

					if (guess_valid) {
						/* Send valid guess signal to active player */
						send_or_die(active_sd, &guess_valid, sizeof(guess_valid), MSG_NOSIGNAL);

						/* Send entire guess to inactive player */
						send_or_die(inactive_sd, &guess_len, sizeof(guess_len), MSG_NOSIGNAL);
						send_or_die(inactive_sd, guess, guess_len, MSG_NOSIGNAL);

						/* Insert guess into the valid guess trie */
						insert_key(guess_trie, guess);
						p1_turn = !p1_turn; // alternate turns
						// continue round
					} else {
						/* Send invalid guess signal to both players */
						send_or_die(active_sd, &guess_valid, sizeof(guess_valid), MSG_NOSIGNAL);
						send_or_die(inactive_sd, &guess_valid, sizeof(guess_valid), MSG_NOSIGNAL);

						(*inactive_score)++;
						break; // end round
					}
				}

				/* Increment round number */
				round++;
			}

			free_trie(&guess_trie);
			/* Don't need to free dict trie here because it was
			   never modified in the child and fork is copy-on-write */

			/* Close client socket descriptors in child */
			close(p1_sd);
			close(p2_sd);

			/* Exit child process */
			exit(0);

		} else if (pid > 0) { // parent
			
			/* Close client socket descriptors */
			close(p1_sd);
			close(p2_sd);

		} else { // error
			fprintf(stderr, "Error: Fork failed\n");
			exit(EXIT_FAILURE);
		}
	}

	/* Don't need to free the trie  because we never get this far and
	   it is used for the duration of the program - better to just 
	   let the OS reclaim the heap space */
}

/* 
 * Generate board_size characters with at least 1 vowel
 * and store them in board
 */ 
void generate_board(char* board, int board_size) {

	/* Seed the PRNG */
	srand(time(NULL));

	bool any_vowel = false; /* Keep track of whether we've generated any vowels */

	/* Generate board_size random letters */
	for(int i = 0; i < board_size; i++){
		char c = (rand() % 26) + 'a';

		/* If current letter is a vowel, set any_vowel flag to true */
		if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
			any_vowel = true;
		}
		board[i] = c;
	}

	/* If none were vowels, set last letter of board to a random vowel */
	if (!any_vowel) {
		char *val = "aeiou";
		board[board_size-1]  = val[rand() % 5];
	}
}

/*
 * Check whether guess is valid - i.e. it meets all of the following criteria:
 * 1) all characters are alphabetic
 * 2) it's in the word list
 * 3) it hasn't been guessed already this round (it's not in the guess_trie)
 * 4) it only contains letters from the game board
 * 5) the frequency of each letter in the guess is less than or equal to the frequency
 *       of that letter in the game board
 */
bool check_guess(char* guess, struct Trie* dict_trie, char* board, int size, struct Trie* guess_trie) {
	
	/* Check whether all chars are alphabetic */
	for (int h = 0; h < strlen(guess); h++) {
		if(tolower(guess[h]) < 'a' || tolower(guess[h]) > 'z') {
			return false;
		}		
	}
	
	/* Check whether word is in the dictionary trie */
	if (!search_key(dict_trie, guess)) {
		return false;
	}
	
	/* Check whether the word has already been guessed */
	if (search_key(guess_trie, guess)) {
		return false;
	}

	/* Check that the guess letter frequency <= board letter frequency
	   for all letters on the board */

	/* Count frequency of each letter in the board */
	int board_freq[26] = {0};
	for (int i = 0; i < size; i++) {
		board_freq[board[i]-'a']++;
	}

	/* Subtract frequency of each letter in the guess */
	for (int j = 0; j < strlen(guess); j++) {
		board_freq[guess[j]-'a']--;

		/* If any < 0, then guess freq > board freq for current letter
		   so the guess is not valid */
		if (board_freq[guess[j]-'a'] < 0) {
			return false;
		}
	}

	/* If we made it this far, the guess must be valid */
	return true;
}
