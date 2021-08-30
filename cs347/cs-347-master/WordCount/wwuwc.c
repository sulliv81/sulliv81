/* Bo Sullivan CSCI 347
 * Dr. Elglaly 
 * wwuwc.c
 *
*/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#define FALSE (0)
#define TRUE  (1)

/* Purpose: To hard code C's build in feature of WC.
 *
 * One main method will track three variables, tot_chars, tot_lines,
 * and tot_words. 
 *
 * Implementation: Tot_chars will be as simple as a while loop that != EOF.
 * Tot_lines also will be easy by adding a supplemental if statemen to check
 * for '\n'
 * And tot words is a bit more challenging, but only in the regards to
 * keeping a helper variable that is set to off (0) and on (1). Using C's
 * built in feature isspace() to check if the current position of getchar()
 * is on a space and actually at the end of a word. If so, we set our helper var more_to_word to off/FALSE and then we increment
 * tot_words and set it back to on/TRUE. If we don't do this, we catch 3 extra
 * words due to -'s and a double space at the bottom with the author.
 *
 * Returns a word count printout similar to c's built in wc feature. 
*/
int main() {
	int tot_chars = 0 ;	/* total characters */
	int tot_lines = 0 ;	/* total lines */
	int tot_words = 0 ;	/* total words */

	int c;
	int more_to_word;
	while ((c = getchar()) != EOF){
	tot_chars++;
		if (isspace(c)){
		more_to_word = FALSE;
		}
		else if (more_to_word == FALSE) {
		tot_words++;
		more_to_word = TRUE;
		}
		if (c == '\n') {
		tot_lines++;
		}
	}
	printf("%d ", tot_lines);
	printf("%d ", tot_words);
	printf("%d\n", tot_chars);

	return 0 ;
}

