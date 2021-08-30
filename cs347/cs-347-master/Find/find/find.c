/* Bo Sullivan
 * CSCI 347 find.c
 * October 8, 2020
 *
 * Implementation of functions that find values in strings.
 *****
 * YOU MAY NOT USE ANY FUNCTIONS FROM <string.h>
 *****
 */

#include <stdlib.h>
#include <stdbool.h>

#include "find.h"

#define NOT_FOUND (-1)	// integer indicator for not found.

/*
 * Return the index of the first occurrence of <ch> in <string>,
 * or (-1) if the <ch> is not in <string>.
 */
int find_ch_index(char string[], char ch) {
	int i;
	int size = sizeof(string) / sizeof(char);
	for (i = 0; i < size; i++) {
		if (string[i] == ch) {
			return i;
		}
	}
	return NOT_FOUND;
}

/*
 * Return a pointer to the first occurrence of <ch> in <string>,
 * or NULL if the <ch> is not in <string>.
 *****
 * YOU MAY *NOT* USE INTEGERS OR ARRAY INDEXING.
 *****
 */
char *find_ch_ptr(char *string, char ch) {

	while(*string != '\0') {
	if (*string == ch)
		return string;
	else string++;
	}
	return NULL;
}

/*
 * Return the index of the first occurrence of any character in <stop>
 * in the given <string>, or (-1) if the <string> contains no character
 * in <stop>.
 */
int find_any_index(char string[], char stop[]){

	size_t string_size = sizeof(string)/sizeof(string[0]);
	size_t stop_size = sizeof(stop)/sizeof(stop[0]);

	int i = 0;
	int j = 0;
	if (string[i] == NULL || stop[j] == NULL) return NOT_FOUND;
	for (i++; i < string_size - 1; i++) {
		for (j++; j < stop_size - 1; j++) {
			if (string[i-1] == stop[j-1]) {
				return i-1;
			}
		}
		j = 0;
	}

	return NOT_FOUND ;	// placeholder
}

/*
 * Return a pointer to the first occurrence of any character in <stop>
 * in the given <string> or NULL if the <string> contains no characters
 * in <stop>.
 *****
 * YOU MAY *NOT* USE INTEGERS OR ARRAY INDEXING.
 *****
 */
char *find_any_ptr(char *string, char* stop) {

	char	*p = stop;

	while(*string != '\0') {
		while (*p != '\0') {
			if (*string == *p) {
				return string;
			}

			else p++;
		}
		p = stop;
		string++;
	}
	return NULL ;	// placeholder
}

/*
 * Return a pointer to the first character of the first occurrence of
 * <substr> in the given <string> or NULL if <substr> is not a substring
 * of <string>.
 * Note: An empty <substr> ("") matches *any* <string> at the <string>'s
 * start.
 *****
 * YOU MAY *NOT* USE INTEGERS OR ARRAY INDEXING.
 *****
 */
char *find_substr(char *string, char* substr) {

	if (string == '\0' || substr == '\0') return '\0';

	return NULL ;	// placeholder
}
