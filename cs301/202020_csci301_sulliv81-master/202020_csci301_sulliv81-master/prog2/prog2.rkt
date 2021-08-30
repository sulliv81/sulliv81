#lang racket

; Pair Programming Assignment 2
; Jeremy Hummel
; Bo Sullivan
; CSCI 301 Spring 2020
; Dr. Brian Hutchinson
; June 01, 2020

; This program analyzes and manipulate grammars of this form: 
; (define G1 '((S) 
;             (a)
;             (((S) (a S) ()))
;             S))
; Where (S) is the variables, (a) is the alphabet/terminals, ((S) (a S) ()) is the rules and S is the start symbol. 
; We are implimenting 6 functions and 1 extra credit function. These functions are as follows:
; get-rules, get-alphabet, get-variables, get-start-symbol, is-formal-grammar?, is-context-free?, is-right-regular? 
; and the extra credit func is: generate-random-string.

; The grammar S--> aS|(empty string)

; original grammars from hutchinson are G1, G2 and G3. The rest are to practice/test methods

(define G1 '((S) 	; 1st element -- variables
             (a)	; 2nd element -- terminals -- alphabet
             (((S) (a S) ()))	; 3rd element -- list of rules
             S))	; 4th element -- start state -- start variable

(define G2 '((S A)	; 1st element -- variables
             (a b c) 	; 2nd element -- terminals -- alphabet
             (((S) (a A) ()) 		; 3rd element -- list of rules
              ((A) (b S) (c)))	; 3rd element -- list of rules
             S))	; 4th element -- start state -- start variable

(define G3 '((S A B C)	; 1st element -- variables
             (a b c d e f) 	; 2nd element -- terminals -- alphabet
             (((S) (A a b c B d e f C) ())  	; 3rd element -- list of rules
              ((A) (a B) (e C) (a))		; 3rd element -- list of rules
              ((B) (b C) (d))			; 3rd element -- list of rules
              ((C) (d) ())) 			; 3rd element -- list of rules
             S))	; 4th element -- start state -- start variable

(define G4 '((S)	; 1st element -- variables
             (a s)	; 2nd element -- terminals -- alphabet
             (((S) (a) ()))	; 3rd element -- list of rules
             S))	; 4th element -- start state -- start variable

; grammar S-->aA|(empty string) aA--> bS|c
(define G5 '((S A) 	; 1st element -- variables
             (a b c)	; 2nd element -- terminals -- alphabet
             (((S) (a S) ())	; 3rd element -- list of rules
              ((A) (b) (c)))	; 3rd element -- list of rules
             S))	 ; 4th element -- start state -- start variable

(define G6 '((S A B C)      ; 1st element -- variables
             (a b c e f)    ; 2nd element -- terminals -- alphabet
             (((S) (A a b c B c e f C) ()) ; 3rd element -- list of rules
              ((A) (a B) (e C) (a))        ; 3rd element -- list of rules
              ((B) (b C) (c))              ; 3rd element -- list of rules
              ((C) (c) ()))                ; 3rd element -- list of rules
             S))                           ; 4th element -- start state -- start variable

(define G7 '((S A B C)      ; 1st element -- variables
             (a b c e f)    ; 2nd element -- terminals -- alphabet
             (((S) (A a b c B c e f C) ()) ; 3rd element -- list of rules
              ((A) (a B) (e C) (a))        ; 3rd element -- list of rules
              ((B) (b C) (d))              ; 3rd element -- list of rules
              ((C) (d) ()))                ; 3rd element -- list of rules
             S))                           ; 4th element -- start state -- start variable

(define G8 '((S)       ; 1st element -- variables
  (a b c)              ; 2nd element -- terminals -- alphabet
  (1 2 3)              ; 3rd element -- list of rules
  T))                  ; 4th element -- start state -- start variable


; For all functions grammar must be valid!
; Racket representation of G of the grammar is valid if the following are true:
; G is a list of length four
; All but the last element of G is itself a list
; The last element of G is not a list
; Each of the rules in G is a list of lists
; Sigma is a summation and intersection of V is not equal to empty set
; S is an element of V

; This is is a helper from Program 1 that was given by Dr. Hutchinson.
; (contains? x S) : Returns true if and only if x ∈ S.
;
; contains? 
; Whether or not an element is contained within a set.
;
; Produces: boolean
;           
; Parameters:
;     x: an element of a Set S.
;     S: a set.
;
; Returns: a boolean value if an element x is contained in set S.
;

(define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))

; this is from Bo Sullivan's Lab 3 which makes use a helper function contains?
; given to us in program 1 from Dr. Hutchinson.
;
; s-in? 
; Has same functionality as contains? Contains? Make use of recursion and a null check
; so it is a safer call to contains? for s-in?
;
; Produces: boolean
;
; Parameters:
;         a: an element of a set.
;         A; A set a.
;
; Returns: A boolean value if an element (a) is contained in a set (A). True if so, false otherwise.
;
;
; (s-in? a A) : Returns true if and only if a ∈ A.
;

(define (s-in? a A)
  (contains? a A))

; This is a helper function from Bo Sullivan's Lab 3to find intersection
; which makes use of another Lab 3 function, s-in?
;
; s-intersect 
; The intersection of two sets.
;
; Produces: list
;           
;
; Parameters:
;          A: one set or list.
;          B: a secondary set or list to check against.
;
; Returns: The intersection of elements between two sets in form of a list.
;

(define (s-intersect A B)
  (filter (lambda (x) (s-in? x A))B))

; This is from Bo Sullivan's Lab 3, which makes use another Lab-3 function, s-in?
; (s-union A B) : Returns the set containg all (and only) the elements in A ∪ B.
;
; s-union
; The union of two sets.
; 
; Produces: list
;           
;
; Parameters:
;        A: a set or list.
;        B: a secondary set or list.
;
; Returns: The union with no repeating elements between two sets A and B.
;

(define (s-union A B)
    (append
     (filter  (lambda (x) (not(s-in? x A)))B)
            
     (filter (lambda (y) (not (s-in? y B)))A)
    (s-intersect A B)))

; getter
; Helper function that will be used to handle null cases or non list references for the get methods in prog2.
;
; Produces: variable at stop pos in list
;           
;
; Parameters:
;      list: the list of rules/variables/alphabet/start-symbol etc to look at.
;      stop: the position to stop at in the list.
;
; Returns: Will return the position to look at for getter methods.
;

(define (getter list stop)
    (cond 
        ((null? list) '())
        ((equal? (length list) stop) (car list))
        (else (getter (cdr list) stop))))

;1.
; get-variables
;
; Produces: The list of variables of a grammar.
;           
;
; Parameters:
;         G: Grammar as parameter.
;
; Returns: The variables (V) of a grammar.
;
;
; Takes a grammar as input and returns the set of variables (i.e. the first element in the grammar 4-tuple).

(define (get-variables G) 
    (getter G 4))

;2.
; get-alphabet
;
; Produces: The list of alphabet of a grammar.
;           
;
; Parameters:
;          G: Grammar as parameter.
;
; Returns: The alphabet of a grammar.
;
;
; Takes a grammar as input and returns the set of terminals (i.e. the second element in the grammar 4-tuple).

(define (get-alphabet G) 
    (getter G 3))

;3.
; get-rules
;
; Produces: The rules, or terminals, of a grammar.
;           
;
; Parameters:
;         G: a grammar as a parameter.
;
; Returns: The rules list of a grammar, or terminals.
;
;
; Takes a grammar as input and returns the set of rules (i.e. the third element in the grammar 4-tuple).

(define (get-rules G) 
    (getter G 2))

;4.
; get-start-symbol
;
; Produces: The start state of a grammar.
;           
;
; Parameters:
;         G: the grammar passed as parameter.
;
; Returns: The starting symbol or state of a grammar.
;
;
; Takes a grammar as input and returns the start symbol (i.e. the fourth element of the grammar 4-tuple).

(define (get-start-symbol G) 
    (getter G 1))

; All but the last element of G is itself a list #2
; FCheck2?
; Checks whether or not the last element of a grammar is itself a list. Renamed to FCheck2? for calling purposes
; within is-formal-grammar?
;
; Produces: bool
;           
;
; Parameters:
;        list: Checks certain lists of a grammar.
;
; Returns: A boolean value returns if get-vars, get-alphabet, and get-rules are lists.
;

(define (FCheck2? list)
  (and (list? (get-variables list))(list? (get-alphabet list)) (list? (get-rules list))))

; The last element of G is not a list #3
; FCheck3?
; Checks whether or not the last element of a grammar is NOT a list. Renamed to FCheck3? for calling purposes
; within is-formal-grammar?
;
; Produces: bool
;           
;
; Parameters:
;        list: Checks the start symbol list of a grammar.
;
; Returns: a boolean value as to whether or not the last item/start-state is NOT a list. True if not, false if so.
;

(define (FCheck3? list)
  (not (list? (get-start-symbol list))))

; Each of rules in G is a list of lists #4
; FCheck4?
; Whether or not that each of the rules/terminals of a grammar is a list of lists. Renamed to FCheck4? for calling purposes
; within is-formal-grammar?
;
; Produces: bool
;           
;
; Parameters:
;      list: the list of rules of a grammar.
;
; Returns:
;     A boolean value of #t if the rules are list of lists, else #f if not.
;

(define (FCheck4? list)
  (cond
    ((null? list) #t)
    ((list? (car list)) (FCheck4-helper (car list) list))
    (else #f)))

; Helps check sublists of rules
; A helper method for is-formal-grammar? which is checking whether or not the sublists of rules are all
; composed of lists.
;
; Produces: bool
;
; Paramaters:
;     list : the subList of a rules list in a grammar.
;
; Returns:
;   returns a boolean value if the sublist of rules list are composed of lists.
;

(define (FCheck4-helper list orig-list)
  (cond
    ((null? list) (FCheck4? (cdr orig-list)))
    ((list? (car list)) (FCheck4-helper (cdr list) orig-list))
    (else #f)))


; vars and alpha intersect is empty/null #5
; FCheck5?
; if the intersect of the grammar variables and alphabet is empty/null. Renamed to FCheck5? for calling
; purposes and clarity in is-formal-grammar?
;
; Produces: bool
;           
;
; Parameters:
;         vars: the variables of a grammar.
;         alpha: the alphabet of a grammar.
;
; Returns: a boolean value of the intersect of alpha and vars. True if null, false otherwise.
;

(define (FCheck5? vars alpha)
  (null? (s-intersect vars alpha)))

; Is start in variables #6
; FCheck6?
; Checks that the start-symbol is in the Variables. Makes use of s-in? Renamed to FCheck6? for clarity and
; calling purposes in is-formal-grammar?
;
; Produces: bool
;           
; Parameters:
;         start: The start-symbol of a grammar.
;         vars: The variables of a grammar.
;
; Returns: a boolean value if the start symbol is in the Variables. True if so, false otherwise.
;
;

(define (FCheck6? start vars)
  (s-in? start vars))

;5.
; is-formal-grammar?
; Takes a grammar as input and returns #t if it is a valid formal grammar. Returns #f otherwise.
;
; Produces: bool
;           
;
; Parameters:
;          G: a grammar
;
; Returns: #t if it is a valid formal grammar. Returns #f otherwise.
;
;
; Takes a grammar as input and returns #t if it is a valid formal grammar. Returns #f otherwise.
; Check if variables/terminals, production rules, and start variable return non-null.
; should return #t/#f.
; Check if grammer 4 tuple has length 4, return #t/#f.
;

(define (is-formal-grammar? G)
   (and (equal? (length G )4) (FCheck2? G) (FCheck3? G) (FCheck4?
        (get-rules G)) (FCheck5? (get-variables G) (get-alphabet G))
        (FCheck6? (get-start-symbol G) (get-variables G))))

; C-Check1
; Check non-terminal length is only 1 and NOT IN alphabet.
; Checks non-terminals are length 1 and not in alphabet.
;
; Produces: bool
;
; Parameters:
;   rules: meant to represent the list of rules in a grammar.
;   vars : The list of vars in a grammar.
;
; Returns:
;     true if non-terminal of length one and is not in alphabet, false otherwise.
;

(define (C-Check1 rules alphabet)
  (cond
    ((null? rules) #t)
    ((and (equal? (length (caar rules))1) (not (s-in? (caaar rules) alphabet))) ; added extra car 5/24
    (C-Check1 (cdr rules) alphabet))
   (else #f)))

; C-Check2
; Check that first element/left most element is an element of the variables.
; Renamed to C-Check2 as a check for context-free-grammar.
;
; Produces: bool
;
; Parameters:
;    rules: meant to represent the list of rules in a grammar.
;    vars : The list of vars in a grammar.
;
; Returns:
;     a boolean value with whether the leftmost rules are an element of the variables.
;

(define (C-Check2 rules vars)
  (cond
    ((null? rules) #t)
    ((s-in? (caaar rules) vars) (C-Check2 (cdr rules) vars))
   (else #f)))

; alpha-vars-union
; A check on the union relationship of a Grammar between /
; alphabet and variables. A helper to check right most bits of rules.
;
; Produces: bool
;
; Parameters:
;   G : A grammar as input.
;
; Returns:
;   A boolean value of union of alphabet and variables of a grammar.
;

(define (alpha-vars-union G)
  (s-union (get-alphabet G) (get-variables G))) ; alphabet change here and vars

; list-iterator
; a check that sublists are in the union of alphabet and vars.
;
; Produces: bool
; 
; Parameters:
;   subList: the list within the list of rules.
;   alphaVarsUnion: the untions of alphabet and vars.
;
; Returns:
;   A boolean value with as to whether or not the list of lists (rules:
;   is within the union of the alphabet).
;

(define (list-iterator subList alphaVarsUnion)
  (cond
    ((null? subList) #t)
    ((s-in? (car subList) alphaVarsUnion) (list-iterator (cdr subList) alphaVarsUnion))
    (else #f)))
    
; in-rules-and-union
; a result as to if the right side of the rules is in the union of the
; variable and alphabet with recursion to handle any size of rules lists.
;
; Produces: bool
;
; Parameters:
;   subList: Meant to be a sublist of the rules, the terminals.
;   G      : A grammar as input.
;
; Returns:
;   Boolean value in which #t is if the terminal rules are in the union of variables and alphabet.
;

(define (in-rules-and-union subList G)
  (cond
    ((null? subList) #t)
    ((list-iterator (car subList) (alpha-vars-union G)) (in-rules-and-union (cdr subList) G))
    (else #f)))

; C-Check3
;
; Checking that right side of rules, recursively (sub-list of rules) is element of union of alphabet and variables
; Renamed to C-Check3 as a check for context-free-grammar.
;
; Produces: bool
;
; Parameters:
;   rules: meant to represent the list of rules in a grammar.
;   G    : a grammar.
;
; Returns:
;     a boolean value with whether the rules are within the union of alpha and variables.
;

(define (C-Check3 rules G)
  (cond
    ((null? rules) #t)
    ((in-rules-and-union (cdar rules) G) (C-Check3 (cdr rules) G))
    (else #f)))

;6.
; is-context-free?
;
; Takes a grammar as input and returns #t if it is a formal grammar and each rule fits the definition of context free
; grammar in the Formal Grammar Guide. Returns #f otherwise.
;
; Produces: bool
;
; Parameters:
;          G: a grammar as input parameter.
;
; Returns: A boolean value of true if is context-free? and is-formal-grammar? false otherwise.
;
;
; Check if S-start is an element of variables -- step 1
; Check the things the first variable is related to
; Check if first element in rules
; Check if t is part of UNION of alphabet and variables
;

(define (is-context-free? G)
  (and (is-formal-grammar? G) (C-Check1 (get-rules G) (get-alphabet G))
       (C-Check2 (get-rules G) (get-variables G)) (C-Check3 (get-rules G) G))) ;; changes here from get- to getR

; ofRulesandOfAlphabet
;
; Produces: The boolean result of the rules being part of alphabet. Helper method for is-context-free?
;           
;
; Parameters:
;          alpha: the alphabet of a grammar.
;          rules: the rules of a grammar.
;          vars: the variables of a grammar.
;
; Returns: A boolean value if the variable is in the alphabet and rules.
;

(define (ofRulesandOfAlphabet alpha rules vars)
  (cond
    ((null? rules) #t)
    ((not(s-in? (caaar rules) vars)) #f) ;not w/in vars, return false
    ((helperInAlphabet alpha (cdar rules)vars) (ofRulesandOfAlphabet alpha (cdr rules) vars))
    (else #f)))

; helperInAlphabet
; Iterate over outside of list and grab right bit to see if it's within the alphabet.
;
; Produces: bool
;           
;
; Parameters:
;        alpha: the alphabet of a grammar.
;        subRules: the inner rules list of a grammar.
;        vars: the variables of a grammar.
;
;
; Returns: a boolean value if the outer list is within the alphabet. True if so, false otherwise.
;

(define (helperInAlphabet alpha subRules vars)
  (cond
    ((null? subRules) #t)
    ((or (v-a (car subRules) alpha)(v-au (car subRules) alpha vars)) (helperInAlphabet alpha (cdr subRules)vars))
    (else #f)))

; v-a
;
; Produces: bool of whether or not v-a is a condition met by context-free where v->a means that the variables is a sublist of the alphabet.
;          And that length of the sublist is 1.
;           
;
; Parameters:
;      subList: the sublist of the alphabet.
;      alpha: the alphabet of a grammar.
;
; Returns: A boolean value as to whether or not v-a is contained as as a sublist of the alphabet. True if so, false otherwise.
;

(define (v-a subList alpha)
  (cond
    ((null? subList) #t)
    ((not(equal? (length subList) 1)) #f)
    ((s-in? (car subList) alpha) #t)
    ;((s-in? (car subList) alpha) (list-iterator2 (cdr subList) alpha))
    (else #f)))

; v-au
;
; Produces: bool of whether or not v-au is a condition met by context ree where v->au implies that the length is 2 and the rules sublist is contained
;           within the alphabet.
;           
;
; Parameters:
;      subList: The inner list of the rules, the terminals.
;      alpha: The alphabet of a grammar.
;      vars: The variables of a grammar.
;
; Returns: A boolean,  length 2 and follows regular grammar rules. True if so, false otherwise.
;

(define (v-au subList alpha vars)
  (cond
    ((null? subList) #t)
    ((not(equal? (length subList) 2)) #f)
    ;(write (car subList))
    ((and(s-in? (car subList) alpha) (s-in? (car (cdr subList)) vars))#t)
    (else #f)))

; is-right-regular?
; Takes a grammar as input and returns #t if it is a context free grammar and each rule fits the definition of right
; regular grammar in the Formal Grammar Guides. Otherwise returns #f.
;
; Produces: bool
;           
;
; Parameters:
;       G: a grammar
;
; Returns: a boolean value of true if is-right-regular and false otherwise.
;
;7.
; Checks v-->a, where a is element of V and a is element of E(terminals/alphabet),
; or
; v --> au, where v,u is element of V and a is element of E(terminals/alphabet),
; or
; v(single Variable from V) --> emptyString, where v is element of V.

(define (is-right-regular? G)
  (and (is-context-free? G) (ofRulesandOfAlphabet (get-alphabet G) (get-rules G) (get-variables G))))

; find-rules
; Given the parameter R passed, it will check it against the rules sublist, inner list, to find if it is contained there and can
; be used for replacement later in the randString
;
; Produces: list
;           
;
; Parameters:
;     rules: The rules of the grammar.
;     R    : R is an argument, the left-most parameter of the rules sublist (inner list). Returns empty list if not present.
;
; Returns: will find the subList of rules for the argument R.
;
;
; will call random select on list this returns
;

(define (find-rules rules R)
  (cond
    ((null? rules) '()) ; not here
    ((equal? (caaar rules) R) (cdar rules))
    (else (find-rules (cdr rules) R))))   

; randRule
;
; Produces: list of find random rule from the terminals which which to rewrite the variables for our generate-random-string method.
         
;
; Parameters:
;     rules: The rules of a grammar.
;     R    : R is an argument, the left-most parameter of the rules sublist (inner list). Returns empty list if not present.
;
; Returns: A random rule from the terminal rules.
;

(define (randRule rules R)
  (getter (find-rules rules R) (random 1 (+ (length (find-rules rules R))1))))

; no-vars
; A check that our final list--or randString we create--does not contain any Variables in it, ie, it was rewritten
; with the possible rewrite rules to avoid using Variables.
;
; Produces: bool
;      
;
; Parameters:
;      vars: The variables within a grammar.
;      list: the list to check against that no V (variables) are contained in the list.

; Returns: A boolean variable checking that no variables are in the final list created.
;

(define (no-vars vars list)
  (cond
    ((null? list) #t)
    ((s-in? (car list) vars) #f)
    (else (no-vars vars (cdr list)))))

; replace
;
; Produces: list with replaced values at a passed position in a list. 
;           
;
; Parameters:
;   list : list to replace value with.
;   pos : positions to replace at.
;   val : value to replace in the list with.
;
; Returns: returns a list with a replaced value at a passed position.
;

(define (replace list pos val)
  (replace-helper 1 pos '() list val))

; replace-helper
; Helper method to help replace method swap values in a list
;
; Produces: List
;
; Parameters:
;    x: starting position.
;   pos: Position to replace at.
;  emptyList: Empty list.
;      list: List we are replacing in.
;      val: Value to replace at.
;
; Returns: Helper list for replace method

(define (replace-helper x pos emptyList list val)
  
    (cond
      ((> x (length list))
       (reverse emptyList))
      ((= x pos)
       (replace-helper (add1 x) pos (cons val emptyList) list val))
      (else
       (replace-helper (add1 x) pos (cons (list-ref list (sub1 x))emptyList) list val))))
  
; get-from-list
;
; Produces: items from a passed list in conjunction with our generate-random-string method.
;           
;
; Parameters:
;    pos : positions to get from.
;    list : the list we will be grabbing from. Helper method.
;
; Returns: Gets something from generic list. 
;

(define (get-from-list pos list)
  (getter list (- (length list) pos)))

; make-string
;
; Produces: list/string for our generate-random-string method.
;           
;
; Parameters:
;      R: R is an argument, the left-most parameter of the rules sublist (inner list). Returns empty list if not present
;      randStringList: the random string list we are creating for generate-random-string.
;
; Returns: returns the starter string for take-all-vars-away helper method.
;
;
; R is starting non-terminal/start point
;

(define (make-string rules R randStringList)
 (if (null? (find-rules rules R)) randStringList
     (let ((selectedString (getter (find-rules rules R) (random 1 (+ (length (find-rules rules R))1)))))
       (cond
         ((null? selectedString) randStringList)
         ((not (null? randStringList) ) (make-string rules (car selectedString) (append selectedString (cdr randStringList))))
         (else (make-string rules (car selectedString) (append selectedString randStringList)))))))

; take-all-vars-away
;
; Produces: list without the Variables of the string by finding rules for V and replacing.
;           
;
; Parameters:
;    vars: variables of the grammar.
;    rules: rules of the grammar.
;    list: with with which to place elements into.
;    pos: position to add to.
;
; Returns: The string as the variables are being replaced by a random rule not containing a V of the Variables list until no Variables left .
;

(define (take-all-vars-away vars rules list pos)
  (cond
    ( (no-vars vars list) list )
    ( (s-in? (get-from-list pos list) vars )
      (take-all-vars-away vars rules (flatten (replace list (add1 pos) (randRule rules (get-from-list pos list))))0))
    ( else (take-all-vars-away vars rules list (add1 pos)))))

;8.
;Optional extra credit
;
; generate-random-string
; This will take a "context-free" grammar G and produce a string in the language, where the string is represented as a
; list of symbols in the alphabet. It should start with the start symbol, and always rewrite the left-most non-terminal,
; randomly picking among the rewrite rules for that non-terminal (see (random n)). It only needs to support context-free
; grammars, if passed a grammar that is not context-free, it should return an empty list.
;
; Produces: list/string in form: '(a b c d...)
;           
; Parameters:
;       G: grammar as input parameter.
;
; Returns: Returns random string with help of several helper methods. The random string, as stated, with take the left-most non-terminal and
;          rewrite it with it's rewrite rules making sure no Variables are in the string. Returns empty list if not context-free, which relies on
;          also being a formal grammar.
;

(define (generate-random-string G)
  (cond
    ((not(is-context-free? G)) '()) ; returns empty string
    (else (take-all-vars-away (get-variables G) (get-rules G) (make-string (get-rules G) (get-start-symbol G)'())0)))) ; returns empty string




