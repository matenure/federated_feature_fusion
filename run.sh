#### Train all local models
main_traffic-la.py --local

######## Global Model - Hard Alignment (P is a permutation matrix)

#MLP no_permute
python -u main_traffic-la.py --MLP

#MLP permute
python -u main_traffic-la.py --MLP --permute

#GNN learn_graph(ICDF)
python -u main_traffic-la.py

#GNN learn_graph(ICDF) permute
python -u main_traffic-la.py --permute

#GNN learn_graph(Gumbel)
python -u main_traffic-la.py --isGumbel

#GNN learn_graph(Gumbel) permute
python -u main_traffic-la.py --isGumbel --permute

#GNN known_graph
python -u main_traffic-la.py --KnownGraph

#GNN known_graph permute
python -u main_traffic-la.py --KnownGraph --permute

#Baseline examples
python -u main_traffic-la_end2end.py --permute
python -u main_traffic-la_transformer.py --MLP
python -u main_traffic-la_fl.py


######## Global Model - Soft Alignment (P is arbitrary. This is the model reported in the main paper)
#Replace "main_traffic-la.py" with "main_traffic-la_soft.py" in the above hard alignment commonds. e.g.

#GNN learn_graph(ICDF) soft alignment (**final model**)##
python -u main_traffic-la_soft.py --permute

# MLP soft alignment
python -u main_traffic-la_soft.py --MLP --permute

#GNN learn_graph(Gumbel) soft alignment
python -u main_traffic-la_soft.py --isGumbel --permute


##If one wants to change other hyperparameters, addtional settings can be added in the cmds as this example: " --seed=42 --epochs=250 --lr=0.001". For traffic data, the default parameter lr=0.01 is generally good.



