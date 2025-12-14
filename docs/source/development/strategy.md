# Strategy

## Random

First randomly selects a checker from all checkers of the player, then randomly selects a legal move for that checker.

## Minimax with Alpha-Beta Pruning

### Evaluation Function

- $score = f_{sn} + f_{dp} + f_{ks} + f_p$
- Solider Number.
    $f_{sn} = c_{w} * N_{w\_solider} + c_{b} * N_{b\_solider}$ , $c_w = 160/8, c_b = -160/16$.
    For black, reverse the sign.
- Direct Path (to escape). For white,

    $$
        f_{dp} = \begin{cases}
        c_{escape}, & n\_path = 1 \\
        \frac{BIG\_NUM}{2}, & n\_path \geq 1
        \end{cases},
        {c}_{escape} = 20
    $$

    For black,

    $$
        f_{dp} = \begin{cases}
        \frac{BIG\_NUM}{4}, & n\_path = 1 \\
        \frac{BIG\_NUM}{2}, & n\_path \geq 1
        \end{cases}
    $$

- King Safety $f_{ks}$. For white, the coefficient is negative, for black, positive.
  - If the king is in the castle, for each adjacent black soldier, -20.
  - If the king is adjacent to the castle, for each adjacent black soldier or camp, -30.
  - If the king is not adjacent to the castle, for each adjacent black soldier or camp, -50.
  - If the king is in these four areas, the FLAT_AREA, the king safety score adds depends on the number of black soldiers in the same area with table below:

    ```python
        area_safety_table = {
            0: 50.0,
            1: 30.0,
            2: 0.0,
            3: -50.0,
            4: -100.0
        }
    ```

    ```{image} /images/flats.png
    :alt: flats
    :width: 40%
    :align: center
    ```
- Position Score $f_p$: precomputed positional score table for both roles.
  - $f_p = \sum_{i,j=0}^{i,j=9}{weight_{ij} * on_{ij}}$

  ```{image} /images/tablut_heatmap.png
  :alt: heatmap
  :align: center
  ```

### Optimizations

- Early Termination
    - makes Alpha-Beta pruning much more efficient, check captures first and eval a BIG_NUMBER to Quick fall
    - For later win solution, eval a number less than BIG_NUMBER but still big, like BIG_NUMBER/2
- Multiprocess Searching
    - send tasks to 4 process to use 4 cores of CPU. (can be the number of cores of the machine)
    - handle timeout easily. But one problem is that sub-tree state is lost if break from timeout with recursion search. To handle timeout more efficiently could implement search functions with loop.
    - default depth is 3.
