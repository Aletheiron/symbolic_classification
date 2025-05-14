Now I want explore this voyage-style symbolic optimization on the classification task.
Main idea of the system:
information passes through node, 
then it suffers changes,
if new version of information makes to grow utility function of the whole system, signal stays in that node during next step, and again some mutations are done to it. We keep using current node until value of the utility function tends to decline.
In case of lowering of the utility function, information string seeks another node.
Now this search of a node is not random like in previous version. Now nodes exist in n-dimension, so they have coordinates. So as data-signal does. Being at a given point, we can calculate a vector of probabilities, which in turn are calculated based on the inverse distance of nodes to the current location. From this array we sample the next node.
Now I want explore this voyage-style symbolic optimization on the classification task.
What about famous Pokémon dataset?
