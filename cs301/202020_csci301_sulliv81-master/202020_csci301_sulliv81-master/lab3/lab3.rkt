; lab 3 racket
; Bo Sullivan
; Spring 2020

; This program will perform a bunch of functions related to sets. Functions such as intersectoins, unions, cartesian products,
; subset, superset, contains/in? etc will be implemented to test sets.

#lang racket
; defining sample lists for testing
(define A '(A B C D E F G E E E))
(define B '(B A C D E F G))
(define F '(A B C D E F))
(define D '(A B C D))
(define C '(E F G X Z Y))

; using contains from program 1 to as a helper function;
;
(define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))

;product and list as set

; will return if two sets are equal--similar to lab 2 truth-table expressions being equal
(define (s-equal? X Y)
  (lambda (x) (contains? x Y)) 
  (equal? (s-size X) (s-size Y)))

; checks if a set contains an element, using contains? for prog 1
; returns true if element is found in A. parameters are a-element and A-list
(define (s-in? a A)
  (contains? a A))

; checks for intersection of 2 sets
; parameters set A and set B.
; returns the intersection
(define (s-intersect A B)
  (filter (lambda (x) (s-in? x A))B))

; returns union of 2 sets
; parements sets A and B
(define (s-union A B)
    (append
     (filter  (lambda (x) (not(s-in? x A)))B)
            
     (filter (lambda (y) (not (s-in? y B)))A)
    (s-intersect A B))
)
  

;return with out element
; Parameters element a and set A
(define (s-remove a A)
  (filter (lambda (x) (not (equal? x a)))A))

; returns s-diff of B from A
; parameters A list and B list
(define (s-diff A B) ;if B contains elements of A, remove them from A
  ;(if (lambda (y) (contains? y A)) (lambda (y) (contains? y B) (s-remove y A)))
  (filter (lambda (x) (not (s-in? x B))) A))

;product in ordered pairs list
; cartesian product, returns paremeters of set A and B in form or ordered pairs
(define (product A B)
  (append (map (lambda (x)
     (map (lambda (y)
        (list x y)) A)) B)))

; s add
; parements element a of List A
; returns a appended to List
(define (s-add a A)
  (if (s-in? a A) A
      (append A (list a))))

; if subset
; parameters 2 lists A and B
; returns if list A is a subset of B
(define (s-subset? A B)
  (equal? (length(filter (lambda (x) (s-in? x B))A))(length A)))

;returns opposite of subset
; parameters 2 lists A and B
; returns if list B is a subset of A
(define (s-superset? A B)
  (not (s-subset? B A)))

; returns size
; parameters List/set A
; returns the .length equivalent of A
(define (s-size A)
  (length A))


(define (L-union A)
    (append
     (filter  (lambda (x) (s-in? x A))A)))

;return list-to-set L
; parameters List L
; returns list without duplicates
; partial credit?? so close to functioning
(define (list-to-set L [K(list)]) 
  (if (equal? (cdr L) '())
          (append (car L) K)
  (list-to-set
   (cdr L)(filter (lambda (x) (not(equal? ( x (car L)))))cdr L))))
  

