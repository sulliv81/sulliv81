;prog1.rkt
;Bo Sullivan
;Spring 2020
;
;This program will work with the idea of a function we have defined named predicate-logic. What predicate-logic is going to do, is that it is going to
;go over our pre-defined list/Universe of A. From this list, we are implementing existential and universal quantifiers on 10 expressions we've created to
;evaluate their truth value and return a boolean value. So, the predicate-logic function should be able to take a Univere (A-- in form of list) and then as the 2nd
;paramter, it checks more lists of either forall or exists and then in the 3rd parameter, checks it against the 10 expressions we implement in this program.
;
; The predicate logic function will make use of tail-recursion to be able to properly check the form of variables in our expressions passed and then be able to determine
; recursively also, which type of quantifier to evaluate with. use of andmap and ormap will be crucial.

#lang racket

;; Useful helper functions

(define (contains? x S)
  (cond ((null? S) false)
        ((equal? x (car S)) true)
        (else (contains? x (cdr S)))))

(define (implies P Q)
  (or (not P) Q))

;; Universe definitions

(define A '(hutchinson hearne fizzano 241 301 474 493 dennis doris spruce redwood rose))

;; Definitions

(define is-faculty '((hutchinson) (hearne) (fizzano)))

(define is-course '((301) (241) (474) (493)))

(define is-student '((dennis) (doris)))

; (a,b,c) means faculty member a teaches course b to student c
(define teaches-to '((hutchinson 301 dennis) (hutchinson 241 doris) (hearne 301 doris) (hearne 493 dennis) (fizzano 474 doris))) 

(define is-person '((hutchinson) (hearne) (fizzano) (dennis) (doris)))

(define is-tree '((spruce) (redwood)))

(define is-plant '((spruce) (redwood) (rose)))

(define taller-than '((redwood spruce) (redwood doris) (redwood dennis) (spruce doris) (spruce dennis) (dennis doris)))

;; Predicate definitions

(define (is-faculty? x)
  (contains? (list x) is-faculty))

(define (is-course? x)
  (contains? (list x) is-course))

(define (is-student? x)
  (contains? (list x) is-student))

(define (teaches-to? prof course student)
  (contains? (list prof course student) teaches-to))

(define (is-person? x)
	(contains? (list x) is-person))

(define (is-tree? x)
  (contains? (list x) is-tree))

(define (is-plant? x)
  (contains? (list x) is-plant))

(define (taller-than? x y)
	(contains? (list x y) taller-than))

(define (new-expr x)
  (implies (is-tree x) (not (is-person x))))

(define (new-expr2 x y z)
  (implies (is-faculty? x)
           (and (is-course? y)
                (is-student? z)
                (teaches-to? x y z))))

(define (new-expr3 x y z)
  (implies (is-faculty? y)
           (and (is-course? z)
                (is-student? z)
                (teaches-to? x y z))))

; This will help us to recursively determine how many elements we need to evaluate in our expression
; base case if the cdr of A is null, cdr being the element to the right
(define (curryHelp function A)
  (if (null? (cdr A))
      (curry function (cdr A))
   (curryHelp function A)) (curry (universe-helper (function (car A))))) ; helper here

; This will help us traverse the Universe of A, recursively
(define(universe-helper function A)
(if (null? (cdr A))
    (function (car A)) (universe-helper function(cdr A))))

;predicate-logic function. If the list contains forall, we want to use andmap
;if our quantlist contains exists, we will want to use ormap.
(define (predicate-logic A quantsList function)
  (if (contains? 'forall quantsList)
  (andmap (curryHelp function A)) (ormap (curryHelp function A))))


 ;(define (curry function)
 ; (curry (curry (curry function) function) function)))



;curry practice methods
;(define (curry1 function)
;  (curry (curry (curry function) 'dennis) 'redwood))


;practice predicate-logic method with no curry
;(define (predicate-logic A quantsList function)
;(if (contains? 'forall quantsList)
;    (andmap (function A)) (ormap (function A))))




;Will declare if being a tree then also a plant or P -> Q
(define (e1 x)
  (implies (is-tree? x)
           (is-plant? x)))

(define statement1 (predicate-logic A (list 'forall) e1))


;will declare if being a plant implies being a tree and tree NOT being taller than the plant
(define (e2 x y)
  (implies (is-plant? x)
           (and (is-tree? x)
                (not (taller-than? y x)))))

(define statement2 (predicate-logic A (list 'forall 'exists) e2))

;will declare a tree is not a tre and not a tree is taller than a tree
(define (e3 x y)
  (and (is-plant? y)
       (not (is-tree? x)
            (not (taller-than? y x)))))

(define statement3 (predicate-logic A (list 'exists 'exists) e3))

;will declare if being a student implies they are a person
(define (e4 x)
  (implies (is-student? x)
           (is-person? x)))

(define statement4 (predicate-logic A (list 'forall) e4))

;will declare if a faculty is also a student
(define (e5 x)
  (and (is-faculty? x)
       (is-student? x)))

(define statement5 (predicate-logic A (list 'exists) e5))

;will declare if a person is a faculty and a student and is also a course where faculty teaches course to student
(define (e6 x y z)
  (and (is-faculty? x)
       (is-student? y)
       (is-course? z)
       (teaches-to x y z)))

(define statement6 (predicate-logic A (list 'exists 'exists 'exists) e6))

; will declare that no faculty member is a student
(define (e7 x y)
  (and (is-faculty? x)
                (is-student? y)))

(define statement7 (predicate-logic A (list 'exists) e7))

; will decalre Every student takes at least one class
(define (e8 x y)
  (implies (is-student? x)
           (is-course? y)))

(define statement8 (predicate-logic A (list 'forall 'exists) e8))

; will declare some plants are tree
(define (e9 x)
  (implies (is-plant? x)
           (is-tree? x)))
(define statement9 (predicate-logic A (list 'exists) e9))

; will declare no plants take a class
(define (e10 x y)
  (not (is-plant? x)
       (is-course? y)))

(define statement10 (predicate-logic A (list 'forall) e10))


