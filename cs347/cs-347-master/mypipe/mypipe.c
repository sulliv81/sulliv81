/*****************************************************************************
*Write a program that executes two commands using a pipe*
*The two commands should be entered by the user as arguments enclosed by " " and separated by |, e.g. ./mypipe "command1 | command2"
*If no arguments are entered by the user, the program will assume command 1 is ls -l and command 2 is sort.
*The correctness of both commands is totally at the discretion of the user                           *
*The program should execute  the commands in pipe and show the output (if any)
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>               /* strsep, etc. */

#define MAX_NUM_ARGS 20           /* Maximum number of arguments allowed */
#define MAX_STR_LEN 200           /* Maximum string length */



int main(int argc, char * argv[])
{
        int fd[2];                 /* Two ends of the pipe */
        char * lhs = NULL;         /* Left hand side command of the pipe */
        char * rhs = NULL;          /* Right hand side command of the pipe */
        char * lhscommand = "ls";  /* Default command name on left hand side of pipe */
        char * rhscommand = "sort"; /* Default command name on right hand side of pipe */
        char * lhsargs[MAX_NUM_ARGS] = { "ls", "-l", NULL };   /* Default LHS args */
        char * rhsargs[MAX_NUM_ARGS] = { "sort", NULL };       /* Default RHS args */



        /*Parse the user input to extract the commands and their arguments*/
        /*Hint: read about strsep(3) */
        char * line = strdup(argv[1]);
        char * token = strsep(&line, "|");
        lhs = token;
        rhs = line;


	// set up to start getting lhs and rhs args
        char delims[2] = " ";
	char str[100];
	char * rTokens;
        rTokens = strtok(str, delims);
	//rhs parsing with strtok
	while( rTokens != '\0') {
		rTokens = strtok(NULL, delims);
	}
	if (rTokens != '\0') {
		rhscommand = rTokens;
	}


	char str2[100];
	char * lTokens;
	lTokens = strtok(str2, delims);
	//lhs parsin with strtok
	while(lTokens != '\0') {
		lTokens = strtok(NULL, delims);
	}
        if (lTokens != '\0') {
                lhscommand = lTokens;
        }
        if (strcmp(rTokens[0], "sort") != 0 || strcmp(lTokens[0], "ls") != 0) {
                printf("Usage:\n./mypipe.c [\"<LHS-command|RHS-command>\"]");
		exit(0);
        }
	

       /* Create the pipe */
        pipe(fd);     /* fd[0] is read end, fd[1] is write end */

      
       /* Do the forking */
        switch ( fork() )
        {
              
               /* The LHS of command 'ls -l|sort' i.e. 'ls' should be
               executed in the child. */

		if (rhs == NULL && lhs == NULL) {
			if (execvp(lhscommand, lhsargs) < 0) exit(1);
			if (execvp(rhscommand, rhsargs) < 0) exit(1);
		}
		if (execvp(token, rTokens) < 0) exit(1);
               /*The RHS of command 'ls -l|sort' i.e. 'sort' should be
                executed in the parent. */
		if (execvp(line, lTokens) < 0) exit(1);
        }

}
