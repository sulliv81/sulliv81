/* CS 347 -- Mini Shell!
 * Original author Phil Nelson 2000
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <ctype.h>
#include "argparse.h"
#include "builtin.h"

#define LINE_LENGTH 1024 //for buf

/* PROTOTYPES */

void processline (char *line);
ssize_t getinput(char** line, size_t* size);

/* main
 * This function is the main entry point to the program.  This is essentially
 * the primary read-eval-print loop of the command interpreter.
 */

int main () {
 char* buf = NULL;
 size_t size = 0;
 ssize_t len;

//Main loop for getting user input
while(1){
  len = getinput(&buf, &size); //Prompt the user for input and store it in buf
  processline(buf);	       //Process user input

}

if(!feof(stdin)) perror("stdin"); //Checks if there was an error in the input after exiting the main loop

  return EXIT_SUCCESS;
}


/* getinput
* line     A pointer to a char* that points at a buffer of size *size or NULL.
* size     The size of the buffer *line or 0 if *line is NULL.
* returns  The length of the string stored in *line.
*
* This function prompts the user for input.  If the input fits in the buffer
* pointed to by *line, the input is placed in *line.  However, if there is not
* enough room in *line, *line is freed and a new buffer of adequate space is
* allocated.  The number of bytes allocated is stored in *size.
*/
ssize_t getinput(char** line, size_t* size) {
  assert(line != NULL && size != NULL);

  errno = 0;
  //print the prompt
  printf("%% ");
  ssize_t len = getline(line, size, stdin); //Prompt the user for input and store the length

  //Reprompts the user if they enter in nothing
  while(len == 1){
    printf("%% ");
    len = getline(line, size, stdin);
  }
  if(len == -1) perror("getline"); //Checks that there's an error in the input

  /* deletes the \n character at the end of each command (on enter key press) */
  if (len > 0 && (*line)[len-1] == '\n'){
    (*line)[len -1] = '\0';
    --len;
  }

  return len;
}


/* processline
 * The parameter line is interpreted as a command name.  This function creates a
 * new process that executes that command.
 * Note the three cases of the switch: fork failed, fork succeeded and this is
 * the child, fork succeeded and this is the parent (see fork(2)).
 * processline only forks when the line is not empty, and the line is not trying to run a built in command
 */
void processline (char *line)
{
 /*check whether line is empty*/
 assert(line != NULL);
  pid_t cpid;
  int   status;
  int argCount;

  //Parse the input from the user
  char** arguments = argparse(line, &argCount);

  /*check whether arguments are builtin commands
   *fork, and execute the commands
   */
  if (arguments[0] == NULL){ //If there is no command, do nothing and return
    return;
  }

  //Check if the input contains a built in command
  int built = builtIn(arguments, argCount);

  //Input contains a command that isn't built in
  if(built == 0) {
    cpid = fork(); //Fork to run the command
    if (cpid < 0) { // fork not successful
      perror ("fork");
      return;
    }
    // Check if child process
    if (cpid == 0) {
      //Execute the command
      execvp (arguments[0], arguments);
      // return not successful
      fprintf(stderr, "%s: no command found.\n", arguments[0]);
      fclose(stdin);
      exit (EXIT_FAILURE);
    }
    // parent waits for child to finish
    if (wait (&status) < 0) {
      // wait not successful
      perror ("wait");
    }

  }

  /* for loop for freeing memory used in argparse() & processLine() */
	for(int i = 0; i < argCount; i++){
		free(arguments[i]);
	}
	free(arguments);
}
