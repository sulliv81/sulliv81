/*
 * Trie implementation from:
 * https://www.techiedelight.com/trie-implementation-insert-search-delete/
 */
 
#ifndef TRIE_H_
#define TRIE_H_

// define character size
#define CHAR_SIZE 26

// A Trie node
struct Trie
{
    int isLeaf;    // 1 when node is a leaf node
    struct Trie* character[CHAR_SIZE];
};

struct Trie* get_new_trie_node();
void insert_key(struct Trie *head, char* str);
int search_key(struct Trie* head, char* str);
int delete_key(struct Trie **curr, char* str);
void make_empty_trie(struct Trie **curr);
void free_trie(struct Trie **curr);

#endif // TRIE_H_