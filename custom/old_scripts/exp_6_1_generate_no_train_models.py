from model_bet import GNN_Bet
import torch

for seed in range(100):

    for model_size in [10000,100000,300000,900000]:

        torch.manual_seed(seed)

        saving_path = f"./models/seeds_analysis/notrained/notrained_{model_size}_size_{seed}_seed"

        #Model parameters
        hidden = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

        torch.save(model.state_dict(), saving_path)