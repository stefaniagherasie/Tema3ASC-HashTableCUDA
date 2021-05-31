# Tema3ASC-HashTableCUDA
[Tema3 - Arhitectura Sistemelor de Calcul] 
Tema presupune implementarea unei structuri de HashTable folosind CUDA, avand ca target GPU Tesla K40 (hp-sl.q).

> Enuntul se gaseste [aici](https://ocw.cs.pub.ro/courses/asc/teme/tema3).

## Descriere 
Structura de date tip HashTable are urmatoarele caracteristici:
- va face maparea (key → value)
- va putea stoca date de tip `int32`, mai mari strict ca 0, atat pentru chei, cat si pentru valori
- va stoca datele în VRAM si va putea sa isi reajusteze dimensiunea cat sa poata acomoda numarul de perechi (key → value), avand un `loadFactor` decent (sloturi ocupate / sloturi disponibile)
- va face update la valoarea unei chei
- va putea intoarce corect, intr-un timp rapid, valoarea corespunzatoare unei chei

Alocarea ai dealocarea memoriei CUDA se va face folosind doar functiile wrapper: `_cudaMalloc`, `_cudaMallocManaged`, `_cudaFree`.

## Organizare
Tema are urmatoarea organizare:
- `gpu_hashtable.hpp`, `gpu_hashtable.cu`
- `test_map.hpp`, `test_map.cu`
- `Makefile`
- `README.md `

## Rulare si Testare
Rularea si testarea se realizeaza pe **cluster** (username@fep.grid.pub.ro), pe coada `hp-sl.q` (sau  `ibm-dp.q`).



