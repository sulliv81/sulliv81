/* Group 7, Project 1
 * CSCI 347, Dr. Elglaly
 * Fall 2020
 *
 * The purpose of builtin.c will be to interact with myshell.c and argparse.c
 * in order to act as a functional shell in which we will be able to print the 
 * current working directory with pwd, change the directory with cd, and also
 * exit the shell with exitProgram.
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
#include "builtin.h"
#include <string.h>
#include <errno.h>

//Prototypes
static void exitProgram(char** args, int argcp);
static void cd(char** args, int argpcp);
static void pwd(char** args, int argcp);

#define BUF_SIZE 1024

/* builtin
 *
 * builtin takes an array of strings-args and an argcount-argcp
 *
 * built in checks each passed argument, if the given command
 * matches one of the built in commands, that command is called and builtin returns 1.
 * If none of the built in commands match the wanted command, builtin returns 0;
  */
int builtIn(char** args, int argcp)
{
  if(strcmp(args[0], "exit") == 0 && argcp > 0){
    exitProgram(args, argcp);
    return 1;
  } else if(strcmp(args[0], "pwd") == 0 && argcp > 0){
    pwd(args, argcp);
    return 1;
  } else if(strcmp(args[0], "cd") == 0 && argcp > 0){
    cd(args, argcp);
    return 1;
  } else{
    return 0;
  }
}
/* exitProgram
 *
 * exitProgram takes a string array-args and an argument count-argcp
 *
 * exitProgram will check initially to see if exit was called with any
 * arguments which determines the return value. If the argcp is greater
 * than 1, then we can check the value being passed with atoi() and
 * then exit with that value.
 */
static void exitProgram(char** args, int argcp)
{
  int count = argcp;
  if(count < 2){
    exit(0);
  }
  int val = atoi(args[1]);
  exit(val);
}
/* pwd
 *
 * pwd takes a string array-args and an argument count-argcp
 *
 * pwd will make use of getcwd() and a char buffer to retrieve the
 * current working directory and to print it out.
 */
static void pwd(char** args, int argcp)
{
  char cwd[BUF_SIZE];

  if(getcwd(cwd, sizeof(cwd)) != NULL){
    printf("%s\n", cwd);
  } else {
    perror("getcwd() error");
  }
}
/* cd
 *
 * cd takes a string array-args and an argument count-argcp
 *
 * cd will be able to change directories like your typical cd would
 * from the command line. This is possible by the chdir() call.
 * It will run the instances of cd with no arguments, or cd with .. or
 * cd with a desired path to enter.
 */
static void cd(char** args, int argcp)
{
  char cwd[BUF_SIZE];
  char* home = "/home";
  if (args[1] == NULL){
    chdir(home);
    //printf("%s\n", getcwd(cwd,sizeof(cwd)));
  } else if (chdir(args[1]) >= 0){
    //printf("%s\n", getcwd(cwd,sizeof(cwd)));
  } else {
   printf("%s: no such directory\n", args[1]);
  }
}

