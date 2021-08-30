#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "argparse.h"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <assert.h>

#define FALSE (0)
#define TRUE  (1)


/* trimEdges trims leading and trailing whitespace and returns a pointer
 * to the updated string */
char* trimEdges(char* line){

	char *endptr;

	/* Trim leading space (walk pointer to first non-whitespace char) */
	while(isspace((unsigned char)*line)){
		line++;
	} 
	/* Case where leading whitespace leads to null terminator */
	if(*line == '\0'){  
		return line;
	}

	/* Trim trailing space (walk pointer to last non-whitespace char) */
	endptr = line + strlen(line) - 1;
	while(endptr > line && isspace((unsigned char)*endptr)){
		endptr--;
	}
	int len = endptr - line + 1;

	/* var to hold updated trimmed string */
	char* retstr = (char*) malloc(len*sizeof(char));

	/* Iterate through line and store trimmed stringin retstr */
	for(int i = 0; i < len; i++){
		retstr[i] = *(line++);
	}
	/* Setting null terminator at end of string */
	retstr[len] = '\0';

	return retstr;
}

/*
 * argCount is a helper function that takes in a String
 * and returns the number of "words" in the string assuming
 * that whitespace is the only possible delimeter.
 */
static int argCount(char* line){

	int numwords = 1; // initialized to one as assert in myshell > getinput ensures at least 1 argument
	line = trimEdges(line); // call to trimEdges, returns pointer to trimmed string
	ssize_t len = strlen(line);

	/* Counting words seperated by whitespace characters */
	for(int i = 0; i < len-1; i++){
		/* Checking that the current character is whitespace and the  
		 * next character is non-whitespace, then numwords += 1 */
		if(isspace(line[i]) && !isspace(line[i+1])){
			numwords++;
		}
	}
	/* Freeing the memory used by malloc in trimEdges*/
	free(line);
	return numwords;
}

/*
* Argparse takes in a String and returns an array of strings from the input.
* The arguments in the String are broken up by whitespaces in the string.
* The count of how many arguments there are is saved in the argcp pointer
*    Note: this method uses malloc, and the return pointer from this function
*        must have its pointers iteratively freed before freeing the return
*        pointer itself
*    Note: argcp > 0 and line == trimEdges(line) should be true before use
*/
char** argparse(char* line, int* argcp)
{
  char **args;
  line = trimEdges(line); // At this point, can assume that !(isspace(line[0]), and no trailing whitespace
  *argcp = argCount(line);
  int arg_index = 0;
  int line_len = strlen(line);
  args = (char **) malloc(*argcp * sizeof(char*));
  

  //This pointer tracks the beginning of a word
  int word_index = 0;

  /* This loop continues until the prev_ptr is the null terminator in order
  *  to save the last arg with no issues
  */
  for(int j = 0; j < line_len; j++)
  {	
    if((!isspace(line[j]) && (isspace(line[j+1]) || line[j+1] == '\0')))
    {
      //The following code is used to save the arg that was found
      int size = j + 1 - word_index;
      int i = 0;
      char current_arg[size];
      args[arg_index] = (char *) malloc((size) * sizeof(char));

      while(word_index != j+1) current_arg[i++] = line[word_index++];
      
	  current_arg[i] = '\0';
      strcpy(args[arg_index],current_arg);
	  arg_index++;
	  
	  // elimante whitespace till next arg
	  while(isspace(line[++word_index]));
	  j = word_index - 1;
    }
  }
  return args;
}

