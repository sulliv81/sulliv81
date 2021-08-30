;Bo Sullivan
;CSCI 301 lab4.rkt
;May 18 2020
;Spring 2020

#lang racket


(define A '(1 2 3))

(define R1 (list (cons 1 2) (cons 2 1) (cons 1 4) ))
(define R2 (list (cons 2 3) (cons 2 7) (cons 1 5) ))


; Parameters element x of set S
; method to see if an element--or pair--is
; in a set
; returns t/f
(define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))

; Paremeters A of set and a Relation R
; Checks that a relation passed onto a set contains
; a mapping of xRx for each element of x in A.
; returns t/f
(define (reflexive? A R)
  (define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))
  (map (lambda (x) (cons x x)) A) ;returns list in one-to-one mapping of each lambda x in list A, with lambda x.
  (andmap (lambda (y)(contains? y R))(map (lambda (x) (cons x x)) A))) ;filter will pull one at a time from L to the function given to filter--lambda y --
                                          ; and will pass it to contains, the lambda.


; Paremeters A of set and a Relation R
; Checks that a relation passed onto a set does NOT contain
; a mapping of xRx for each element of x in A.
; returns t/f
(define (irreflexive? A R)
  (not (reflexive? A R)))

; Paremeters A of set and a Relation R
; Checks that every element xRy also contains yRx
; returns t/f
(define (symmetric? A R)
    (define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))
  (map (lambda (x) (cons (cdr x) (car x)))R)
  (andmap (lambda (y) (contains?  y R)) (map (lambda (x) (cons (cdr x) (car x)))R)))

; Paremeters A of set and a Relation R
; Checks that every element xRy is NOT contained in yRx
; returns t/f
(define (anti-symmetric? A R)
      (define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))
  (map (lambda (x) (cons (cdr x) (car x)))R)
  (andmap (lambda (y) (not (contains?  y R))) (map (lambda (x) (cons (cdr x) (car x)))R)))


; Paremeters A of set and a Relation R
; Checks that every element xRy or yRx mapping with andmap and or with helper contains?
; Also used example of recursion list pairs from 5.18 racket demo with Alex Ayala to help generate all possible listings
; returns t/f
(define (total? A R) ;make list of things that should be in, then use contains? to see if in
   (define (recursionList L1 L2 [results '()][L2-copy L2])
  (cond
    [(null? L1) results]
    [(null? L2) (recursionList (cdr L1) L2-copy results L2-copy)]
    [else (recursionList L1 (cdr L2) (cons (cons (car L1) (car L2)) results) L2-copy)]))
  ; 1, 2 cdr x and car x will flip
  (ormap (lambda (x) (or (contains?  x R) (contains? (cons (cdr x) (car x)) R))) (recursionList A A)))


; Parameters A set A and relation R
; Also used example of recursion list pairs from 5.18 racket demo with Alex Ayala
; Checks for a transitive relation on a set Similar to total but will need andmap instead of or
; return t/f
(define (transitive? A R)
  (define (recursionList L1 L2 [results '()][L2-copy L2])
  (cond
    [(null? L1) results]
    [(null? L2) (recursionList (cdr L1) L2-copy results L2-copy)]
    [else (recursionList L1 (cdr L2) (cons (cons (car L1) (car L2)) results) L2-copy)]))
  (andmap (lambda (x) (or (contains?  x R) (contains? (cons (cdr x) (car x)) R))) (recursionList A A)))

; parameters A set and 2 relations R1 and R2
;
; A Lot of mentor help with compose as of 5/18
; the idea was needing to use nested recursion to plot values against each other from one relation
; to the other.
; Makes use of recursion and a nested helper fuctions to check coordinates of x and y value
; and then map them to their corresponding values in R2 from R1. For example if R1 contained:
; {(0,1),(1,2),(2,3)} and R2 contained: {(1,4),(2,5),(3,6)} the relation composition would return
; {(0,4),(1,5),(2,6)} .
; will return ordered pairs, no repeats
(define (compose A R1 R2 [results '()])
  (define (helper x-coor y-coor R2 results) ; comparing in r2 ; results storing
    (cond
      [(empty? R2) results]
      [(equal? y-coor (caar R2)) (helper x-coor y-coor (cdr R2) (append results (list(cons x-coor (cdr (car R2))))))]
      [else (helper x-coor y-coor (cdr R2) results)]))
  (cond
    [(empty? R1) results]
    [else (compose A (cdr R1) R2 (helper (car (car R1)) (cdr (car R1)) R2 results))]))

      



 