import numpy as np
import pandas as pd
import torch
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset import get_dataset, get_trajectories
from options import get_options
from utils import CustomLoss

from model import DLinear, NLinear, Linear

args = get_options()
if args.model_name == 'DLinear':
    model = DLinear(args.Lag, args.Horizon, args.kernel_size, args.individual, args.enc_in).to(args.device)
elif args.model_name == 'NLinear':
    model = NLinear(args.Lag, args.Horizon, args.individual, args.enc_in).to(args.device)
elif args.model_name == 'Linear':
    model = Linear(args.Lag, args.Horizon, args.individual, args.enc_in).to(args.device)
else:
    raise NotImplementedError

results = []
Ex, Ey, Er = 40, 400, 50
trajectories = get_trajectories()
for id, trajectory in trajectories:
    # reset_model(model)
    
    train_loader, inputs, targets, init_value, x_scaler, y_scaler, enter_index, leave_index =\
        get_dataset(trajectory, Ex, Ey, Er, args.stride, args.Lag, args.Horizon, args.batch_size)

    # criterion = nn.MSELoss().to(args.device)
    criterion = CustomLoss().to(args.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_loss = 0.0

        model.train()  # switch model to train mode
        for data, target in train_loader:
            data, target = data.to(args.device), target.to(args.device)
            # Clear the gradients of all optimized variables
            model.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        lr_scheduler.step()

        if epoch % 100 == 0:  # print every 100 epochs
            print(f'Epoch: {epoch}/{args.epochs}\tTrain Loss: {train_loss:.6f}\t')


    model.eval()  # switch model to evaluation mode

    with torch.no_grad():
        inputs[:, 0] = x_scaler.transform(inputs[:, 0].reshape(-1, 1)).flatten()
        inputs[:, 1] = y_scaler.transform(inputs[:, 1].reshape(-1, 1)).flatten()
        inputs = torch.from_numpy(inputs).unsqueeze(0).to(args.device).float()
        outputs = np.squeeze(model(inputs).detach().cpu().numpy(), axis=0)

    # outputs = outputs[0:targets.shape[0], :]
    outputs[:, 0] = x_scaler.inverse_transform(outputs[:, 0].reshape(-1, 1)).flatten()
    outputs[:, 1] = y_scaler.inverse_transform(outputs[:, 1].reshape(-1, 1)).flatten()

    outputs = np.cumsum(outputs, axis=0) + init_value
    # targets = np.cumsum(targets, axis=0) + init_value  
    # Calculate MSE and MAE
    # mse = mean_squared_error(outputs, targets)
    # mae = mean_absolute_error(outputs, targets)

    # print(f'Test MSE: {mse:.6f}')
    # print(f'Test MAE: {mae:.6f}')

    distances = np.linalg.norm(outputs - np.array([Ex, Ey]), axis=1)
    inside_circle_indices = np.where(distances <= Er)[0]
    pred_index = inside_circle_indices[-1]
    pred_index = enter_index + pred_index * args.stride
    print(f'diff time: {pred_index - leave_index}')

    results.append({
        'New_ID': int(id),
        'enter_index': enter_index,
        'leave_index': leave_index,
        'predict_index': pred_index
    })

df = pd.DataFrame(results)
df.to_csv(f'./data/{args.model_name}_predictions.csv', index=False)