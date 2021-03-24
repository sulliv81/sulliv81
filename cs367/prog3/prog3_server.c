/* demo_server.c - code for example client program that uses TCP 
*
* Code based off Brian Hutchinson's demo_server.c
*
* Bo Sullivan
* prog3_server.c
* Winter 2021 CSC 367
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
#include <ctype.h>
#include <time.h>
#define QLEN 6 /* size of request queue */

const time_t timeoutLen = 60;

/* client struct to store connection information pertaining to clients.*/
uint8_t nameRecSize;
uint16_t msgRecSize;

int participant_fds[255] = { -1 };
int observer_fds[255] = { -1 };

//just using a master struct idk.
typedef struct client {
	time_t timeoutLength;
	char userName[10];
	int active;
	int sd;
	int obsd;
    bool attached;
}client;

typedef struct timeouts {
	time_t timeoutLength;
	int sd;
}timeouts;
struct timeouts ttv[510];
/*Up to 255 participants*/
struct client participantClient[255];
/*Up to 255 observers*/
struct client observerClient[255];


//all file descriptors for sockets
fd_set tot_fd;
/* update_times takes a timeout struct, start and end time_t
*  Returns: void
* Intended use is updating time in the ttv time struct.
*/
void update_times(timeouts *t, time_t start, time_t end) {
	int index;
	int lowest = timeoutLen;

	for (int i = 0; i < 510; i++) {

		t[i].timeoutLength = difftime(end, start);
	}
}
/* get_timeout_time takes a timeout struct
* returns lowest time to use to calculate which time to call select with
*/
int get_timeout_time(timeouts *t) {
	int lowest = timeoutLen;

	for (int i = 0; i < 510; i++) {

		while (lowest > 0 && t[i].timeoutLength > 0) {
			if (t[i].timeoutLength < lowest) {
				lowest = t[i].timeoutLength;
			}
			break;
		}
		break;
	}
	return lowest;
}
/* update_times takes a timeout struct, start and end time_t
*  Returns: int of timeout index
* Intended use is getting index to find sd to disconnect.
*/
int get_timeout_time_index(timeouts *t) {
	int index = 0;
	int lowest = timeoutLen;

	for (int i = 0; i < 510; i++) {

		while (lowest > 0 && t[i].timeoutLength > 0) {
			if (t[i].timeoutLength < lowest) {
				lowest = t[i].timeoutLength;
				index = i;
			}
			break;
		}
		break;
	}
	return index;
}
//deactivate timeouts
void timeoutZero(timeouts *t, int index) {
    t[index].timeoutLength = 0;
}
//60 seconds back on the clock!
void resetTimeout(timeouts *t, int index) {
    t[index].timeoutLength = 60;
}
/* clear_entry takes a client struct *t and  int index
*  Returns void.
*  Intended use is upon a client disconnecting, updating the respective struct accordingly.
*  Boolea flag denotes if the disconnect is a participant.
*/
void clear_entry(client *t, int index) {
	
	memset(t[index].userName, 0, sizeof(t[index].userName));
	t[index].active = 0;
	t[index].sd = -1;
	t[index].obsd = -1;
	t[index].attached = false;

}
/* is_full takes an int *a (respective socket descriptor array[])
*  Returns a boolean denoting if given descriptor array is full or not
*  Intended use is checking if we can allow up to 255 connections.
*/
bool is_full(int *a) {
	for (int i = 0; i < 255; i++) {
		if (a[i] == 0) {
			return false;
		}
	}
	return true;
}
/* find_sd takes an int sd
*  Returns the socket descriptor given the existing connection
*  Intended use is checking existing connections to process after select
*/
int find_sd(int sd) {
    for (int i = 0; i < 255; i++) {
        if (participant_fds[i] == sd) {
            return participant_fds[i];
        }
    }
    for (int i = 0; i < 255; i++) {
        if (observer_fds[i] == sd) {
            return observer_fds[i];
        }
    }
}
/* find_sd takes an int sd
*  Returns the socket descriptor given the existing connection
*  Intended use is checking existing connections to process after select
*/
int find_sd_index(int sd) {
    for (int i = 0; i < 255; i++) {
        if (participant_fds[i] == sd) {
            return i;
        }
    }
    for (int i = 0; i < 255; i++) {
        if (observer_fds[i] == sd) {
            return i;
        }
    }
}
int find_struct_index(int sd) {
	for (int i = 0; i < 255; i++) {
		if (participantClient[i].sd == sd) {
			return i;
		}
	}
	for (int i = 0; i < 255; i++) {
		if (observerClient[i].sd == sd) {
			return i;
		}
	}
}
/* obs_or_participant takes an int sd
*  Returns a 1 indicating if the socket is a participant or 2 for observer
*  Intended use is checking existing connections to process after select
*/
int obs_or_participant(int sd) {
    for (int i = 0; i < 255; i++) {
        if (participant_fds[i] == sd) {
            return 1;
        }
    }
    for (int i = 0; i < 255; i++) {
        if (observer_fds[i] == sd) {
            return 2;
        }
    }
}

//here are my thoughts for fnding the smallest timeout. For loop to find smallest index in struct.
// this returns index of lowest timeoutLength, call select on.
int getNextTimeout(client *t) {
	time_t lowest;
	time_t stop;
	int sd;
	stop = 60;
	sd = 0;
	for (int i = 0; i < 510; i++) {
		if (t[i].timeoutLength < stop) {
			lowest = t[i].timeoutLength;
			sd = t[i].sd;
		}
	}
	return sd;
}
void stopTimeout(client *t, int sd) {
	t[sd].timeoutLength = 0;
}

void startTimeout(client *t, int sd) {
	t[sd].timeoutLength = 60;
}
/* find_participant_obs_sd takes a char* name, client *t, and size_t nameLen
* Returns the socket descriptor for a participant with an affiliated observer
* Intended use for finding attached observer socket descriptors to send to.
*/
int find_participant_obs_sd(char* name, client *t, size_t nameLen) {
    for (int i = 0; i < 255; i++) {
        //pass empty spots in struct without seg faulting
        if (t[i].userName == NULL) {
            continue;
        }
        if (strncmp(name, t[i].userName, nameLen) == 0)  {
			if (t[i].attached == true) {
            	return t[i].obsd;
			}

        } 
    }
    return 0;
}
/* int sendPrivate takes char* name, char* msg, size_t msgLen 
*  Returns an int. One indicating success, 0 indicating failure
*  Intended use is sending private messages to affiliated observers
*/
int sendPrivate(char* name, char* msg, size_t msgLen) {
	int r, targetSD;

	targetSD = find_participant_obs_sd(name, participantClient, strlen(name));
	uint16_t host2Net = htons(msgLen);

	r = send(targetSD, &host2Net, 2, MSG_NOSIGNAL);
	r = send(targetSD, &msg, msgLen, MSG_NOSIGNAL);
}
/* send_all_fds takes a socket descriptor set (denoting observers), char* msg and size_t msgLen
*  Returns an int: 1-indicates success, 0-indicates failure.
*  Intended use for sending public messages to all active Observers
*/
int send_all_fds(int sds[], char* msg, size_t msgLen) {
	int r;
	msg[msgLen] = '\0';
	size_t host2Net = htons(msgLen);
	for (int i = 0; i < 255; i++) {
		if (sds[i] != 0 && sds[i] != -1) {

			r = send(sds[i], &host2Net, 2, MSG_NOSIGNAL);
			//send Msg next
			r = send(sds[i], msg, msgLen, MSG_NOSIGNAL);

			if (r < 0) {
				return 0;
			}
			return 1;
		}
	}
	memset(msg, 0, sizeof(msg));
	return 0;
}
/* send_all_fds_name takes a client struct *o (denoting observer), char* name and size_t nameLen
*  Returns an int: 1-indicates success, 0-indicates failure.
*  Intended use for notifying all observers a new user has joined they could affiliate with.
*/
int send_all_fds_name(client *o, char* name, size_t nameLen) {
	int r;

    char* user = "User ";
    char* joined = " has joined";
    size_t len = 26 - (10 - nameLen);
    char msgOut[len+1];
	msgOut[len] = '\0';
    strcat(msgOut, user);
    strcat(msgOut, name);
    strcat(msgOut, joined);
	uint16_t host2Net = htons(len);

	for (int i = 0; i < 255; i++) {
		if (o[i].active > 0) {

            
			r = send(o[i].sd, &host2Net, 2, MSG_NOSIGNAL);
			//send Msg next and not host2net on second call.
			r = send(o[i].sd, msgOut, len, MSG_NOSIGNAL);

			if (r < 0) {
				return 0;
			}
			return 1;
		}
	}
	memset(msgOut, 0, sizeof(msgOut));
	return 0;
}
/* handle_disconnection takes an int sd.
* Returns: void
* This method is going to help reflect disconnects and send message
* If someone disconnects, we need to check if observer or participant.
* This method is kind of beefy. Anytime recv returns 0, this will get called.
* It checks, given a socket if it is participant or client.  It then checks if
* participant, if it has an observer. It then applies protocol and also sends the
* user has left message to active observer fds. Closes sockets and clears fd_set.
*/
void handle_disconnection(int sd) {
	
	int index, sd_type, obs_sd, obsIndex, structIndex, obsStructIndex;
	

	//finds sd index in array of sds
	index = find_sd_index(sd);
	sd_type = obs_or_participant(sd);

	if (sd_type == 1) {
		
		structIndex = find_struct_index(sd);
			
		if (participantClient[structIndex].active == 1) {
			char* name;
			name = participantClient[structIndex].userName;
			char* user = "User ";
			char* left = " has left";
			size_t len = 26 - (10 - strlen(name));
			char nameOut[len-1];
			memset(nameOut, 0, sizeof(nameOut));

			strcat(nameOut, user);
			strcat(nameOut, name);
			strcat(nameOut, left);
			
			send_all_fds(observer_fds, nameOut, len);
			memset(nameOut, 0, sizeof(nameOut));

			if (participantClient[structIndex].attached == true){
				obs_sd = participantClient[structIndex].obsd;
				obsStructIndex = find_struct_index(obs_sd);
				obsIndex = find_sd_index(obs_sd);
				participant_fds[index] = -1;
				observer_fds[obsIndex] = -1;
				clear_entry(participantClient, structIndex);
				clear_entry(observerClient, obsStructIndex);
				close(sd);
				FD_CLR(sd, &tot_fd);
				close(obs_sd);
				FD_CLR(obs_sd, &tot_fd);
				return;
			}
			else {
				participant_fds[index] = -1;
				clear_entry(participantClient, structIndex);
				close(sd);
				FD_CLR(sd, &tot_fd);
				return;
			}
		}
		else {
			participant_fds[index] = -1;
			close(sd);
			FD_CLR(sd, &tot_fd);
			return;
		}
	}
	//Observer, update to no longer have affiliation though for participant
	else if (sd_type == 2) {
		obsStructIndex = find_struct_index(obs_sd);

		if (observerClient[obsStructIndex].active == 1) {
			observer_fds[index] = -1;
			clear_entry(observerClient, obsStructIndex);
			close(sd);
			FD_CLR(sd, &tot_fd);
			return;
		}
		else {
			close(sd);
			FD_CLR(sd, &tot_fd);
			return;
		}
	}
	else {
		close(sd);
		FD_CLR(sd, &tot_fd);
		return;
	}
}
/* find_participant(char* name, client *t, and size_t nameLen
*  Returns a boolean value denoting if participant was found.
*  Intended use is for observers looking up names and participants
*  trying to connect and see if a name exists.
*/
bool find_participant(char* name, client *t, size_t nameLen) {
    for (int i = 0; i < 255; i++) {

        if (t[i].userName == NULL) {
           continue;
        }
		//need dynamic sizing based on name from recv call.
        if (strncmp(name, t[i].userName, nameLen) == 0)  {
            return true;
        }   
    }
    return false;
}
/* attach_observer takes a int observerSD, char* name, client *t, and size_t nameLen
*  Returns void.
*  Intended Use is if has_observer returns false, call attach_observer method.
*/
void attach_observer(int observerSD, char* name, client *t, size_t nameLen) {
    for (int i = 0; i < 255; i++) {

        if (t[i].userName == NULL) {
           continue;
        }
		//need dynamic sizing based on name from recv call.
        if (strncmp(name, t[i].userName, nameLen) == 0)  {
            t[i].attached = true;
			t[i].obsd = observerSD;
            return;
        }   
    }
    return;
}
/* has_observer takes a char* name, client struct, and size_t nameLen
*  Returns: Boolean.
*  Intended Use: Figure out if a observer is trying to connect to a 
*  participant that already has on observer.
*/ 
bool has_observer(char* name, client *t, size_t nameLen) {
    for (int i = 0; i < 255; i++) {

        if (t[i].userName == NULL) {
           continue;
		   
        }
		//need dynamic sizing based on name from recv call.
        if (strncmp(name, t[i].userName, nameLen) == 0)  {
            if (t[i].attached == true) {
                return true;
            }
        }   
    }
    return false;
}
//not sure if I will need this one as my struct should also have the sd with it.
int find_participant_sd(char* name, client *t) {
    for (int i = 0; i < 255; i++) {
        //pass empty spots in struct without seg faulting
        if (t[i].userName == NULL) {
            continue;
        }
        if (strncmp(name, t[i].userName, 10) ==0)  {
            return t[i].sd;
        } 
    }
    return 0;
}

/* find_participant_active takes a char *name, client *t, and size_t nameLen
*  Returns a boolean. True if the participant is active, false if not, or name not found.
*  Intended use is making sure participant is active before sending in handle_msg().
*/
bool find_participant_active(char *name, client *t, size_t nameLen) {

	bool b = find_participant(name, t, nameLen);

	if (b) {
		for (int i = 0; i < 255; i++) {
			//pass empty spots in struct without seg faulting
			if (t[i].active == 1) {
				return true;
			}
			return false;
		}
	}
	else {
		return false;
	}
    
}

/* handle_msg takes an int sd, client struct *t, and char* uname
*  Returns an int: 1 indicates success, 0 indicates disconnect, -1 indicates failure
*  Intended use, handle incoming messages from participant and depending on message type'
*  private or public, handle them accordingly and send.
*/
int handle_msg(int sd, client *t, char* uname) {
	
	//maybe a loop to check if active b4 doign this?
	int n;
    uint16_t msgRecLen;
	uint16_t msgOutLen;
	char* pound = "#";
    char* space = " ";
	char* colon = ":";
	char* priv = "@";
	char* arrow = ">";
	char* warning = "Warning: user $";
	char* dne = " doesn't exist. . .";
	uint8_t nameLen = strlen(uname);

	int spaces = 11 - nameLen;

	
	n = recv(sd, &msgRecLen, sizeof(msgRecLen), MSG_NOSIGNAL);
	if (n == 0) {handle_disconnection(sd); return 0;}
	//trying this and htink update paritcipants_fds[i] = 0;
	msgRecLen = ntohs(msgRecLen);

	//msgLen > 1024 = close socket
    if (msgRecLen > 1024) {
        handle_disconnection(sd);
        return -1;
    }

	//rec msg i think error is here
	char msg[1025];//  = {'\0'} ;
	memset(msg, 0, sizeof(msg));
	n = recv(sd, &msg, msgRecLen, MSG_NOSIGNAL);
	if (n == 0) {handle_disconnection(sd); return 0;}

	//need to find name
	char outmsg[1024];
	memset(outmsg, 0, 1024);
	//priv msg--need to find sd and name. and parse until space.
	if (msg[0] == '@') {
		//parse the input...
		char* token = strtok(msg, "@");
		char* targetName = strtok(token, " ");

		if (strlen(targetName) > 10 || strlen(targetName) == 0) {
			handle_disconnection(sd);
			return 0;
		}
		
		bool b = find_participant_active(targetName, t, strlen(targetName));
		if (b) {

			strcat(outmsg, pound);

			for (int i = 0; i < spaces; i++) {
				strcat(outmsg, space);
			}
			strcat(outmsg, targetName);
			strcat(outmsg, colon);
			strcat(outmsg, space);

			sendPrivate(targetName, outmsg, strlen(outmsg));
            memset(outmsg, 0, sizeof(outmsg));
			memset(msg, 0, sizeof(msg));

			return 1;
		}
		//targetName does not exist.
		else {
			printf("Could not find participant.\n");

			strcat(outmsg, warning);
			strcat(outmsg, targetName);
			strcat(outmsg, dne);

			msgOutLen = strlen(outmsg);
			uint16_t htonsLen = htons(msgOutLen);
			send(sd, &htonsLen, sizeof(msgOutLen), MSG_NOSIGNAL);
			send(sd,  outmsg, msgOutLen, MSG_NOSIGNAL);
            memset(outmsg, 0, sizeof(outmsg));
			memset(msg, 0, sizeof(msg));
			return 0;
		}
	}
	//public msg
	else {
		strcat(outmsg, arrow);

		for (int i = 0; i < spaces; i++) {
			strcat(outmsg, space);
		}
		strcat(outmsg, uname);
		strcat(outmsg, colon);
		strcat(outmsg, space);
		strcat(outmsg, msg);

		//send to all
		msgOutLen = strlen(outmsg);
		send_all_fds(observer_fds, outmsg, msgOutLen);
        memset(outmsg, 0, sizeof(outmsg));
		memset(msg, 0, sizeof(msg));
		return 1;

	}
	memset(outmsg, 0, sizeof(outmsg));
	memset(msg, 0, sizeof(msg));
    return 1;
}

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
	int sd, sd2, sd3; /* socket descriptors */
	int participantSocket;
	int observerSocket;
	int port;
	uint16_t part_port;
	uint16_t obs_port;
	int alen; /* length of address */
	int optval = 1; /* boolean value when we set socket option */
	char buf[1024]; /* buffer for string the server sends */
	uint8_t nameSize;


	//set to -1 value for meaning no connection/socket descriptor

	//thinking per Dr. Hutchinson, timestruct array.
	int timeOuts[510] = { 0 };

	if( argc != 3 ) {
		fprintf(stderr,"Error: Wrong number of arguments\n");
		fprintf(stderr,"usage:\n");
		fprintf(stderr,"./server server_port server_port\n");
		exit(EXIT_FAILURE);
	}

	memset((char *)&sad,0,sizeof(sad)); /* clear sockaddr structure */
	sad.sin_family = AF_INET; /* set family to Internet */
	sad.sin_addr.s_addr = INADDR_ANY; /* set the local IP address */

	//creat participant port
	part_port = atoi(argv[1]); /* convert argument to binary */
	if (part_port > 0) { /* test for illegal value */
		sad.sin_port = htons(part_port);
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
	participantSocket = socket(PF_INET, SOCK_STREAM, ptrp->p_proto);
	if (participantSocket < 0) {
		fprintf(stderr, "Error: Socket creation failed\n");
		exit(EXIT_FAILURE);
	}

	/* Allow reuse of port - avoid "Bind failed" issues */
	if( setsockopt(participantSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0 ) {
		fprintf(stderr, "Error Setting socket option failed\n");
		exit(EXIT_FAILURE);
	}

	/* Bind a local address to the socket */
	if (bind(participantSocket, (struct sockaddr *)&sad, sizeof(sad)) < 0) {
		fprintf(stderr,"Error: Bind failed\n");
		exit(EXIT_FAILURE);
	}

	/* Specify size of request queue */
	if (listen(participantSocket, QLEN) < 0) {
		fprintf(stderr,"Error: Listen failed\n");
		exit(EXIT_FAILURE);
	}
	
	//create observer port
	memset((char *)&cad,0,sizeof(cad)); /* clear sockaddr structure */
	cad.sin_family = AF_INET; /* set family to Internet */
	cad.sin_addr.s_addr = INADDR_ANY; /* set the local IP address */
	obs_port = atoi(argv[2]); /* convert argument to binary */
	if (obs_port > 0) { /* test for illegal value */
		cad.sin_port = htons(obs_port);
	} else { /* print error message and exit */
		fprintf(stderr,"Error: Bad port number %s\n",argv[2]);
		exit(EXIT_FAILURE);
	}

	/* Map TCP transport protocol name to protocol number */
	if ( ((long int)(ptrp = getprotobyname("tcp"))) == 0) {
		fprintf(stderr, "Error: Cannot map \"tcp\" to protocol number");
		exit(EXIT_FAILURE);
	}

	/* Create a socket */
	observerSocket = socket(PF_INET, SOCK_STREAM, ptrp->p_proto);
	if (observerSocket < 0) {
		fprintf(stderr, "Error: Socket creation failed\n");
		exit(EXIT_FAILURE);
	}

	/* Allow reuse of port - avoid "Bind failed" issues */
	if( setsockopt(observerSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0 ) {
		fprintf(stderr, "Error Setting socket option failed\n");
		exit(EXIT_FAILURE);
	}

	/* Bind a local address to the socket */
	if (bind(observerSocket, (struct sockaddr *)&cad, sizeof(cad)) < 0) {
		fprintf(stderr,"Error: Bind failed\n");
		exit(EXIT_FAILURE);
	}

	/* Specify size of request queue */
	if (listen(observerSocket, QLEN) < 0) {
		fprintf(stderr,"Error: Listen failed\n");
		exit(EXIT_FAILURE);
	}
	
	int total_fds;
	FD_ZERO(&tot_fd);

	FD_SET(participantSocket, &tot_fd);
	FD_SET(observerSocket, &tot_fd);
	total_fds = participantSocket+1;

	int ttv_counter = 0;

	/* Main server loop - accept and handle requests */
	while (1) {

		fd_set ready_to_read;
		memcpy(&ready_to_read, &tot_fd, sizeof(fd_set));
		int n, current_socket, timeout_socket, timeout_index;
		char yess = 'Y';
		char noo = 'N';
		char taken = 'T';
		char invalid = 'I';

		time_t the_timeout;
		time_t now;
		time(&now);
		the_timeout = get_timeout_time(ttv);
		struct timeval tv;
		tv.tv_sec = the_timeout;
		tv.tv_usec = 0;
		

		n = select(total_fds + 1, &ready_to_read, NULL, NULL, &tv);

		time_t stop;
		time(&stop);
		update_times(ttv, stop, now);

		if (n < 0) {
			fprintf(stderr, "select failed\n");
			exit(EXIT_FAILURE);
		}
		//this is where the timoeouts happen n == 0
		else if (n == 0) {
			timeout_index = get_timeout_time_index(ttv);
			timeout_socket = ttv[timeout_index].sd;
			handle_disconnection(timeout_socket);
			ttv[timeout_index].sd = -1;
			ttv[timeout_index].timeoutLength = 0;
			ttv_counter -= 1;
			continue;
		}
        else {
            //new port socket
            if (FD_ISSET(participantSocket,  &ready_to_read)) {

                current_socket = accept(participantSocket, (struct sockaddr *)&sad, &alen);
                bool f = is_full(participant_fds);
                
                if (f) {send(current_socket, &noo, sizeof(noo), MSG_NOSIGNAL); close(current_socket); }
                else { 
                    ttv[ttv_counter].sd = current_socket;
					ttv[ttv_counter].timeoutLength = timeoutLen;
					ttv_counter += 1;

                    for (int i = 0; i < 255; i++) {
                        if (participant_fds[i] < 1) {
                            participant_fds[i] = current_socket;
                            break;
                        }	
                    }
                    send(current_socket, &yess, sizeof(yess), MSG_NOSIGNAL);
                    FD_SET(current_socket, &tot_fd);
                    total_fds = current_socket > total_fds ? current_socket : total_fds;
                }
                continue;
            }
            //observer socket connections
            else if (FD_ISSET(observerSocket,  &ready_to_read)) {
                current_socket = accept(observerSocket, (struct sockaddr *)&cad, &alen);

				bool g = is_full(observer_fds);

				if (g) {send(current_socket, &noo, sizeof(noo), MSG_NOSIGNAL); close(current_socket);}
				else { 

					ttv[ttv_counter].sd = current_socket;
					ttv[ttv_counter].timeoutLength = timeoutLen;
					ttv_counter += 1;
					FD_SET(current_socket, &tot_fd);

					char yess = 'Y';
					send(current_socket, &yess, sizeof(yess), MSG_NOSIGNAL);

					total_fds = current_socket > total_fds ? current_socket : total_fds;

					for (int i = 0; i < 255; i++) {
						if (observer_fds[i] == 0) {
							observer_fds[i] = current_socket;
							break;
						}
					}
				}
                continue;
            }
            else {
                //else another established client participant
                for (int i = 0; i <= total_fds; i++) {
                   //now find if participant or observer?
                    if(FD_ISSET(i, &ready_to_read)) {
                        current_socket = find_sd(i);
                        int x;
                        x = obs_or_participant(current_socket);

                        int n;
                        if (x == 1) {
                            if (participantClient[i].active == 0) {
                                if ((n = recv(current_socket, &nameRecSize, sizeof(nameRecSize), MSG_NOSIGNAL)) == 0) {
									handle_disconnection(current_socket);
                                }
                                else {
                                    n = recv(current_socket, buf, nameRecSize,  MSG_NOSIGNAL);
									if (n == 0) {handle_disconnection(current_socket);}

                                    if (nameRecSize < 1 || nameRecSize > 10) {
                                        send(current_socket, &invalid, sizeof(invalid), MSG_NOSIGNAL);
                                        continue;
                                    }
                                    char* nameCheck;
                                    nameCheck = buf;
                                    bool b = find_participant(nameCheck, participantClient, nameRecSize);
                                    if (b == 1) { 
                                        send(current_socket, &taken, sizeof(taken), MSG_NOSIGNAL);
										memset(nameCheck, 0, sizeof(nameCheck));
                                        memset(buf, 0, sizeof(buf));
                                        continue;
                                    }
                                    else {
                                        send(current_socket, &yess, sizeof(yess), MSG_NOSIGNAL);
                                        strcpy(participantClient[i].userName, nameCheck);
                                        participantClient[i].active = 1;
                                        participantClient[i].sd = current_socket;
                                        send_all_fds_name(observerClient, nameCheck, nameRecSize);
										memset(nameCheck, 0, sizeof(nameCheck));
                                        memset(buf, 0, sizeof(buf));
                                        continue;
                                    }
                                }
                            }
                            else {
                                handle_msg(current_socket, participantClient, participantClient[i].userName);
                                continue;
                            }
                        }
                        else if (x == 2) {
                            if (observerClient[i].active < 1) {

                                if ((n = recv(current_socket, &nameRecSize, sizeof(nameRecSize), MSG_NOSIGNAL)) == 0) {
									handle_disconnection(current_socket);
                                }
                                else {
                                    n = recv(current_socket, buf, nameRecSize, MSG_NOSIGNAL);
									if (n == 0) {handle_disconnection(current_socket);}

                                    if (nameRecSize < 1 || nameRecSize >  10) {
                                        send(current_socket, &invalid, sizeof(invalid), MSG_NOSIGNAL);
										
                                        continue;
                                    }
                                    char* nameCheck2;
                                    nameCheck2 = buf;
                                    bool b = find_participant(nameCheck2, participantClient, nameRecSize);

                                    if (b == 1) {
                                        /*If name exists...need to affiliate so we can send.*/
                                        observerClient[i].sd = current_socket;
                                        observerClient[i].active = 1;
                                        bool bb = has_observer(nameCheck2, participantClient, nameRecSize);

                                        if (!bb) {
                                            attach_observer(current_socket, nameCheck2, participantClient, nameRecSize);
                                            send(current_socket, &yess, sizeof(yess), 0);
											memset(nameCheck2, 0, sizeof(nameCheck2));
                                            memset(buf, 0, sizeof(buf));
                                            continue;
                                        }
                                        else {
                                            send(current_socket, &taken, sizeof(taken), MSG_NOSIGNAL);
											memset(nameCheck2, 0, sizeof(nameCheck2));
                                            memset(buf, 0, sizeof(buf));
                                            continue;
                                        }
                                    }
                                    else {
                                        send(current_socket, &noo, sizeof(noo), 0);
                                        close(current_socket);
                                        FD_CLR(current_socket, &tot_fd);
                                        observer_fds[i] = -1;
										memset(nameCheck2, 0, sizeof(nameCheck2));
                                        memset(buf, 0, sizeof(buf));
                                        continue;
                                    }
                                }
                            }
                            else if (observerClient[i].active == 1) {
                                
                            }
                        }
                    }
                }
                continue;
            }
		}
        continue;
	} // while loop
}
