# Tema3ASC-HashTableCUDA
[Tema3 - Arhitectura Sistemelor de Calcul] <br> 
Tema presupune implementarea unei structuri de HashTable folosind CUDA, avand ca target GPU Tesla K40 (hp-sl.q).<br>
Enuntul se gaseste [aici](https://ocw.cs.pub.ro/courses/asc/teme/tema3).

## Descriere 
Structura de date tip HashTable are urmatoarele caracteristici:
- va face maparea (key → value)
- va putea stoca date de tip `int32`, mai mari strict ca 0, atat pentru chei, cat si pentru valori
- va stoca datele în VRAM si va putea sa isi reajusteze dimensiunea cat sa poata acomoda numarul de perechi (key → value), avand un `loadFactor` decent (sloturi ocupate / sloturi disponibile)
- va face update la valoarea unei chei
- va putea intoarce corect, intr-un timp rapid, valoarea corespunzatoare unei chei

Alocarea si dealocarea memoriei CUDA se va face folosind doar functiile wrapper: `_cudaMalloc`, `_cudaMallocManaged`, `_cudaFree`.

## Organizare
Tema are urmatoarea organizare:
- `gpu_hashtable.hpp`, `gpu_hashtable.cu`
- `test_map.hpp`, `test_map.cu`
- `Makefile`
- `README.md `

## Rulare si Testare
Rularea si testarea se realizeaza pe **cluster** (username@fep.grid.pub.ro), pe coada `hp-sl.q` (sau  `ibm-dp.q`).

Testarea manuala / automata se face cu:
```shell
    ./gpu_hashtable <entries> <chunks> <speed>
    python3 bench.py
```

## Implementare
Am realizat 2 structuri (`HashNode` si `HashTable`) in care sa stochez 
datele primite. Functia de hash este de forma `((value * a) / b) % limit`  unde a si b sunt
niste numere prime alese random.

- `insertBatch()` - apeleaza functia `kernel_insertBatch()` care calculeaza 
hashcode pentru fiecare cheie. Daca locul din HashTable indicat de hashcode este
ocupat, se cauta la dreapta urmatoarea pozitie libera. Se foloseste functia 
`atomicExchange` pentru a evita probleme de concurenta. Se verifica daca structura
de HashTable are nevoie de reshape. 
- `reshape()` - se copiaza toate valorile din tabela initiala intr-o tabela de
dimensiunea data. La inserarea in noua tabela se calculeaza din nou hashcode-ul.
- `getBatch()` - se cauta in tabela valorile corespunzatoare cheilor folosind
hashcode-ul. Valorile intoarse de functia `kernel_getBatch()` se copiaza intr-un 
vector de pe host.

## Bibliografie
1. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicexch
2. https://ocw.cs.pub.ro/courses/asc/laboratoare/07
3. https://ocw.cs.pub.ro/courses/asc/laboratoare/08
4. https://ocw.cs.pub.ro/courses/asc/laboratoare/09


