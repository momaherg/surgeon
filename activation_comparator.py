
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.stats import zscore


class ActivationComparator:
    """Compare and visualize activations between target and trial runs."""

    def __init__(self):
        self.comparison_results = None

    def compute_similarity_metrics(
        self,
        target_activations: Dict,
        trial_activations: List[Tuple[str, Dict]],
        normalize: bool = True,
        use_last_token_only: bool = True
    ) -> pd.DataFrame:
        """
        Compute dot products and L2 distances between target and trial activations.

        Args:
            target_activations: Dictionary of target activations from extract_model_activations
            trial_activations: List of (label, activations_dict) tuples
            normalize: Whether to normalize activations before comparison
            use_last_token_only: Whether to use only the last token's activation

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        # Get common layers between target and all trials
        target_layers = set(target_activations.keys())
        common_layers = target_layers.copy()
        for _, trial_acts in trial_activations:
            common_layers &= set(trial_acts.keys())

        # Sort layers for consistent ordering
        layer_order = sorted(common_layers, key=lambda x: (
            0 if 'embedding' in x.lower() else
            1 if 'layer_' in x else
            2 if 'hidden_state_' in x else
            3 if 'final' in x else
            4 if 'lm_head' in x else 5,
            int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
        ))

        for layer_name in layer_order:
            # Skip if layer doesn't exist in target
            if layer_name not in target_activations:
                continue

            target_act = target_activations[layer_name]

            # Handle different tensor shapes
            if isinstance(target_act, torch.Tensor):
                target_act = target_act.float()

                # Use only last token if specified
                if use_last_token_only and target_act.dim() >= 2:
                    # For shape (batch, seq, hidden) or (seq, hidden)
                    if target_act.dim() == 3:
                        target_act = target_act[:, -1, :].squeeze()
                    elif target_act.dim() == 2:
                        target_act = target_act[-1, :].squeeze()

                # Flatten to 1D for comparison
                target_flat = target_act.flatten()

                if normalize:
                    norm = torch.norm(target_flat)
                    if norm > 0:
                        target_flat = target_flat / norm
            else:
                continue

            row_data = {'layer': layer_name}

            for trial_label, trial_acts in trial_activations:
                if layer_name not in trial_acts:
                    row_data[f'{trial_label}_dot'] = np.nan
                    row_data[f'{trial_label}_l2'] = np.nan
                    continue

                trial_act = trial_acts[layer_name]

                if isinstance(trial_act, torch.Tensor):
                    trial_act = trial_act.float()

                    # Use only last token if specified
                    if use_last_token_only and trial_act.dim() >= 2:
                        if trial_act.dim() == 3:
                            trial_act = trial_act[:, -1, :].squeeze()
                        elif trial_act.dim() == 2:
                            trial_act = trial_act[-1, :].squeeze()

                    # Flatten to 1D
                    trial_flat = trial_act.flatten()

                    # Ensure same size
                    min_size = min(target_flat.shape[0], trial_flat.shape[0])
                    target_truncated = target_flat[:min_size]
                    trial_truncated = trial_flat[:min_size]

                    if normalize:
                        norm = torch.norm(trial_truncated)
                        if norm > 0:
                            trial_truncated = trial_truncated / norm

                    # Compute metrics
                    dot_product = torch.dot(target_truncated, trial_truncated).item()
                    l2_distance = torch.norm(target_truncated - trial_truncated).item()

                    row_data[f'{trial_label}_dot'] = dot_product
                    row_data[f'{trial_label}_l2'] = l2_distance
                else:
                    row_data[f'{trial_label}_dot'] = np.nan
                    row_data[f'{trial_label}_l2'] = np.nan

            results.append(row_data)

        self.comparison_results = pd.DataFrame(results)
        return self.comparison_results

    def create_comparison_heatmap(
        self,
        target_activations: Dict,
        trial_activations: List[Tuple[str, Dict]],
        normalize: bool = True,
        use_last_token_only: bool = True,
        title: str = "Activation Similarity Analysis",
        height: int = 800,
        show_values: bool = True
    ) -> go.Figure:
        """
        Create an interactive heatmap visualization comparing activations.

        Args:
            target_activations: Target activation dictionary
            trial_activations: List of (label, activations_dict) tuples
            normalize: Whether to normalize before comparison
            use_last_token_only: Use only last token
            title: Title for the visualization
            height: Height of the figure
            show_values: Whether to show values in cells

        Returns:
            Plotly figure object
        """
        # Compute similarity metrics
        df = self.compute_similarity_metrics(
            target_activations,
            trial_activations,
            normalize,
            use_last_token_only
        )

        # Prepare data for heatmaps
        trial_labels = [label for label, _ in trial_activations]

        # Create matrices for dot products and L2 distances
        dot_matrix = []
        l2_matrix = []

        for _, row in df.iterrows():
            dot_row = [row.get(f'{label}_dot', np.nan) for label in trial_labels]
            l2_row = [row.get(f'{label}_l2', np.nan) for label in trial_labels]
            dot_matrix.append(dot_row)
            l2_matrix.append(l2_row)

        dot_matrix = np.array(dot_matrix)
        l2_matrix = np.array(l2_matrix)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Dot Product (Higher = More Similar)",
                          "L2 Distance (Lower = More Similar)"),
            horizontal_spacing=0.12
        )

        # Create hover text
        hover_dot = []
        hover_l2 = []
        for i, layer in enumerate(df['layer']):
            dot_row_hover = []
            l2_row_hover = []
            for j, label in enumerate(trial_labels):
                dot_val = dot_matrix[i, j]
                l2_val = l2_matrix[i, j]
                dot_row_hover.append(f"Layer: {layer}<br>Trial: {label}<br>Dot Product: {dot_val:.4f}")
                l2_row_hover.append(f"Layer: {layer}<br>Trial: {label}<br>L2 Distance: {l2_val:.4f}")
            hover_dot.append(dot_row_hover)
            hover_l2.append(l2_row_hover)

        # Dot product heatmap (higher is better - use Greens)
        text_dot = [[f"{val:.3f}" if not np.isnan(val) else ""
                    for val in row] for row in dot_matrix] if show_values else None

        fig.add_trace(
            go.Heatmap(
                z=dot_matrix,
                x=trial_labels,
                y=df['layer'].tolist(),
                colorscale='RdYlGn',  # Red-Yellow-Green
                zmid=0.5 if normalize else 0,
                text=text_dot,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertext=hover_dot,
                hoverinfo='text',
                colorbar=dict(
                    title="Dot<br>Product",
                    x=0.45
                )
            ),
            row=1, col=1
        )

        # L2 distance heatmap (lower is better - reverse Greens)
        text_l2 = [[f"{val:.3f}" if not np.isnan(val) else ""
                   for val in row] for row in l2_matrix] if show_values else None

        fig.add_trace(
            go.Heatmap(
                z=l2_matrix,
                x=trial_labels,
                y=df['layer'].tolist(),
                colorscale='RdYlGn_r',  # Reversed: Green for low values
                text=text_l2,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertext=hover_l2,
                hoverinfo='text',
                colorbar=dict(
                    title="L2<br>Distance",
                    x=1.02
                )
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            height=max(height, 400 + len(df) * 20),
            showlegend=False,
            font=dict(size=11)
        )

        # Update axes
        fig.update_xaxes(title_text="Trials", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Trials", row=1, col=2, tickangle=45)
        fig.update_yaxes(title_text="Layers", row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=2)

        return fig

    def create_summary_visualization(
        self,
        target_activations: Dict,
        trial_activations: List[Tuple[str, Dict]],
        normalize: bool = True,
        use_last_token_only: bool = True,
        top_k_layers: int = 10
    ) -> go.Figure:
        """
        Create a summary visualization showing most and least similar layers.

        Args:
            target_activations: Target activation dictionary
            trial_activations: List of (label, activations_dict) tuples
            normalize: Whether to normalize before comparison
            use_last_token_only: Use only last token
            top_k_layers: Number of top/bottom layers to show

        Returns:
            Plotly figure with summary statistics
        """
        # Compute metrics
        df = self.compute_similarity_metrics(
            target_activations,
            trial_activations,
            normalize,
            use_last_token_only
        )

        trial_labels = [label for label, _ in trial_activations]

        # Compute average similarity per layer
        avg_dot = df[[f'{label}_dot' for label in trial_labels]].mean(axis=1)
        avg_l2 = df[[f'{label}_l2' for label in trial_labels]].mean(axis=1)

        # Create summary dataframe
        summary_df = pd.DataFrame({
            'layer': df['layer'],
            'avg_dot_product': avg_dot,
            'avg_l2_distance': avg_l2,
            'dot_std': df[[f'{label}_dot' for label in trial_labels]].std(axis=1),
            'l2_std': df[[f'{label}_l2' for label in trial_labels]].std(axis=1)
        })

        # Sort by average dot product
        summary_df = summary_df.sort_values('avg_dot_product', ascending=False)

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Top {top_k_layers} Most Similar Layers (by Dot Product)",
                f"Top {top_k_layers} Most Different Layers (by Dot Product)",
                f"Top {top_k_layers} Closest Layers (by L2 Distance)",
                f"Top {top_k_layers} Farthest Layers (by L2 Distance)"
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Top similar by dot product
        top_similar = summary_df.nlargest(top_k_layers, 'avg_dot_product')
        fig.add_trace(
            go.Bar(
                x=top_similar['avg_dot_product'],
                y=top_similar['layer'],
                orientation='h',
                marker_color='green',
                error_x=dict(array=top_similar['dot_std'], visible=True),
                text=[f"{v:.3f}" for v in top_similar['avg_dot_product']],
                textposition='outside'
            ),
            row=1, col=1
        )

        # Most different by dot product
        most_different = summary_df.nsmallest(top_k_layers, 'avg_dot_product')
        fig.add_trace(
            go.Bar(
                x=most_different['avg_dot_product'],
                y=most_different['layer'],
                orientation='h',
                marker_color='red',
                error_x=dict(array=most_different['dot_std'], visible=True),
                text=[f"{v:.3f}" for v in most_different['avg_dot_product']],
                textposition='outside'
            ),
            row=1, col=2
        )

        # Closest by L2
        closest_l2 = summary_df.nsmallest(top_k_layers, 'avg_l2_distance')
        fig.add_trace(
            go.Bar(
                x=closest_l2['avg_l2_distance'],
                y=closest_l2['layer'],
                orientation='h',
                marker_color='green',
                error_x=dict(array=closest_l2['l2_std'], visible=True),
                text=[f"{v:.3f}" for v in closest_l2['avg_l2_distance']],
                textposition='outside'
            ),
            row=2, col=1
        )

        # Farthest by L2
        farthest_l2 = summary_df.nlargest(top_k_layers, 'avg_l2_distance')
        fig.add_trace(
            go.Bar(
                x=farthest_l2['avg_l2_distance'],
                y=farthest_l2['layer'],
                orientation='h',
                marker_color='red',
                error_x=dict(array=farthest_l2['l2_std'], visible=True),
                text=[f"{v:.3f}" for v in farthest_l2['avg_l2_distance']],
                textposition='outside'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="Layer Similarity Summary Across All Trials",
            height=800,
            showlegend=False
        )

        fig.update_xaxes(title_text="Avg Dot Product", row=1, col=1)
        fig.update_xaxes(title_text="Avg Dot Product", row=1, col=2)
        fig.update_xaxes(title_text="Avg L2 Distance", row=2, col=1)
        fig.update_xaxes(title_text="Avg L2 Distance", row=2, col=2)

        return fig


def visualize_activation_comparison(
    target_result: Dict,
    trial_results: List[Tuple[str, Dict]],
    normalize: bool = True,
    use_last_token_only: bool = True,
    save_html: Optional[str] = None
) -> Tuple[go.Figure, go.Figure]:
    """
    Main function to visualize activation comparisons.

    Args:
        target_result: Result from extract_model_activations for target
        trial_results: List of (label, result_dict) tuples for trials
        normalize: Whether to normalize activations
        use_last_token_only: Whether to use only last token
        save_html: Optional path to save HTML files

    Returns:
        Tuple of (heatmap_figure, summary_figure)
    """
    comparator = ActivationComparator()

    # Extract activations from results
    target_activations = target_result['activations']
    trial_activations = [(label, result['activations']) for label, result in trial_results]

    # Create visualizations
    heatmap_fig = comparator.create_comparison_heatmap(
        target_activations,
        trial_activations,
        normalize=normalize,
        use_last_token_only=use_last_token_only,
        title="Activation Similarity: Target vs Trials",
        show_values=True
    )

    summary_fig = comparator.create_summary_visualization(
        target_activations,
        trial_activations,
        normalize=normalize,
        use_last_token_only=use_last_token_only,
        top_k_layers=10
    )

    # Save if requested
    if save_html:
        heatmap_fig.write_html(f"{save_html}_heatmap.html")
        summary_fig.write_html(f"{save_html}_summary.html")
        print(f"Saved visualizations to {save_html}_heatmap.html and {save_html}_summary.html")

    # Display figures
    heatmap_fig.show()
    summary_fig.show()

    return heatmap_fig, summary_fig
