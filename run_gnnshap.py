import argparse
import pickle
import time
import os

import torch
from tqdm.auto import tqdm

from dataset.utils import get_model_data_config
from gnnshap.explainer import GNNShapExplainer

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset.configs import get_config

from mymodels import GCN, GAT
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--result_path', type=str, default=None,
                        help=('Path to save the results. It will be saved in the config results '
                              'path if not provided.'))
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to use for GNNShap')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--sampler', type=str, default='GNNShapSampler',
                        help='Sampler to use for sampling coalitions',
                        choices=['GNNShapSampler', 'SVXSampler', 'SHAPSampler',
                                'SHAPUniqueSampler'])
    parser.add_argument('--solver', type=str, default='WLSSolver',
                        help='Solver to use for solving SVX', choices=['WLSSolver', 'WLRSolver'])
    
    # SVXSampler maximum size of coalitions to sample from
    parser.add_argument('--size_lim', type=int, default=3)

    args = parser.parse_args()

    dataset_name = args.dataset
    num_samples = args.num_samples
    batch_size = args.batch_size
    sampler_name = args.sampler
    solver_name = args.solver


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name != 'Elliptic':
        model, data, config = get_model_data_config(dataset_name, load_pretrained=True, device=device)
    else:
        config = get_config(dataset_name)
        root_path = config['root_path']

        # Load the dataset
        data = torch.load(os.path.join(config['root_path'], 'data.pt')) 
        model = torch.load(os.path.join(config['root_path'], 'model.pt'))
        model = model.to(device)
    

    # Ensure test_nodes is set
    if 'test_nodes' not in config or config['test_nodes'] is None:
        config['test_nodes'] = data.test_mask.nonzero(as_tuple=True)[0].tolist()


    test_nodes = config['test_nodes']

    result_path = args.result_path if args.result_path is not None else config["results_path"]


    if sampler_name == "SVXSampler":
        extra_param_suffixes = f"_{args.size_lim}"
    else:
        extra_param_suffixes = ""

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        _, pred = out.max(dim=1)

        # Extract actual labels and predictions
        actuals = data.y[data.test_mask].cpu().numpy()
        predictions = pred[data.test_mask].cpu().numpy()

        # Calculate accuracy
        correct = int((predictions == actuals).sum())
        acc = correct / len(actuals)
        print(f'Test Loss: {test_loss.item()}, Accuracy: {acc}')

        # Calculate recall
        recall = recall_score(actuals, predictions, average='macro')  # 'macro' averages recall across classes
        print(f'Recall: {recall}')

        # Calculate precision
        precision = precision_score(actuals, predictions, average='macro')  # 'macro' averages precision across classes
        print(f'Precision: {precision}')

        # Calculate F1 score
        f1 = f1_score(actuals, predictions, average='macro')  # 'macro' averages F1 score across classes
        print(f'F1 Score: {f1}')

        # Compute confusion matrix
        cm = confusion_matrix(actuals, predictions)
        print(f'Confusion Matrix:\n{cm}')

        # Plot confusion matrix
        class_labels = ['Non-Fraud', 'Fraud']  # Update with your actual class labels
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()


    #explain_node_idx = 0
    for r in range(args.repeat):
        results = []
        
        shap = GNNShapExplainer(model, data, nhops=config['num_hops'], verbose=0, device=device,
                           progress_hide=True)
        start_time = time.time()

        failed_indices = []
        for ind in tqdm(test_nodes, desc=f"GNNShap explanations - run{r+1}"):
            try:
                explanation = shap.explain(ind, nsamples=num_samples,
                                            sampler_name=sampler_name, batch_size=batch_size,
                                            solver_name=solver_name, size_lim=args.size_lim)
                results.append(explanation.result2dict())
            except RuntimeError as e:
                failed_indices.append(ind)
                if 'out of memory' in str(e):
                    print(f"Node {ind} has failed: out of memory")
                else:
                    print(f"Node {ind} has failed: {e}")
            except Exception as e:
                print(f"Node {ind} has failed. General error: {e}")
                failed_indices.append(ind)

        rfile = (f'{result_path}/{dataset_name}_GNNShap_{sampler_name}_{solver_name}_'
                   f'{num_samples}_{batch_size}{extra_param_suffixes}_run{r+1}.pkl')
        with open(rfile, 'wb') as pkl_file:
            pickle.dump([results, 0], pkl_file)
        
        if len(failed_indices) > 0:
            print(f"Failed indices: {failed_indices}")
