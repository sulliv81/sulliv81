#lang racket
;Bo Sullivan
;Spring 2020, WWU, CSCI 301, Professor Hutchinson
;lab2

; will evaluate expression (P and Q or R)
(define (example-expr1 P Q R)
  (or (and P Q) R))

; will evaluate expression not P or (Q ^ not R) ^ not S
(define (example-expr2 P Q R S)
  (or (not P)
      (and (and Q (not R))
           (not S))))
; guessing this is how to call this
;(example-expr1 #t #f #t)

(define (permutations size)
  (let ((elements (list #t #f)))
    (if (zero? size)
        '(())
        {append-map (lambda (p)
                      (map (lambda (e) (cons e p)) elements))
                    (permutations (sub1 size))})))
;(permutations 1)
;(permutations 2)

; will define P -> Q
(define (implies P Q)
  (or (not P)
      (or Q)))
  
;(implies #t #f)

; will define P -> (Q->R)
(define (expr1 P Q R)
  (or (not P)
      (or (not Q)
          (or R))))

;practice for expr1
;(expr1 #f #t #t)

; will define Q -> (P -> R)
(define (expr2 P Q R)
  (or (not Q)
      (or (not P)
          (or R))))

;practice for expr2
;(expr2 #t #t #f)

; will define R->(Q->P)
(define (expr3 P Q R)
  (or (not R)
      (or (not Q)
          (or P))))

;(expr3 #f #t #t)

; will define P ^ A
(define (expr4 P Q)
  (and P Q))

;(expr4 #f #f)

; will define P or ~Q
(define (expr5 P Q)
  (or P (not Q)))

;(expr5 #f #t)

; will define an equivalent pair with expr7 P ^ (Q or R)
(define (expr6 P Q R)
  (and  (or Q R) P))
       
;(expr6 #f #t #t)

; will define an equivalent pair with expr 6 which is (P ^ Q) or (P ^R)
(define (expr7 P Q R)
  (or (and P Q) (and P R)))

;(expr7 #t #t #f)

; will define (P ^ Q) or (R ^ S)
(define (expr8 P Q R S)
  (or (and P Q) (and R S)))


;(expr8 #f #f #f #t)

; will define (P or Q) or (P or S)
(define (expr9 P Q R S)
  (or (or P Q) (or R S)))

;(expr9 #f #t #f #f)


; using permutations function to get all possible combinations for expressiong passed, returned as list of list
; passing list into map, then defining new lambda expression that takes in one input and will pass into lambda
; function, then using apply on that inner list
(define (truth-table expr)
    (map (lambda (x) (apply expr x)) (permutations (procedure-arity expr))))

; check 2 lists for equality
  
(define (equivalent? expression1 expression2)
  (equal? (truth-table expression1) (truth-table expression2)))

(equivalent? expr6 expr7)


