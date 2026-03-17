#!/usr/bin/env python3
"""
Visualization Module for Contaminant Transport Analysis
Comprehensive Visualization Tools for Simulation and ML Results

Description:
    This module provides comprehensive visualization tools for analyzing results
    from numerical simulations and machine learning models in contaminant transport
    studies. It offers a complete suite of static and dynamic visualization
    capabilities for environmental risk assessment applications.
    
    Key functionalities:
    - Concentration field visualization (contour plots, heatmaps)
    - Temporal evolution animations and videos
    - Risk level mapping and classification visualization
    - Machine learning model performance comparison
    - Confusion matrix analysis and metrics visualization
    - Feature importance plotting
    - Snapshot generation at specific time points
    - Interactive dashboards for comprehensive analysis

All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    Secretary of Science, Humanities, Technology and Innovation, SECIHTI (Secretaria de Ciencia, Humanidades, Tecnología e Innovación). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
    SIIIA-MATH: Soluciones de Ingeniería. México

Date:
    February, 2025.

Last Modification:
    August, 2025.
"""

# Standard libraries
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

class ContaminantVisualizer:
    """
    Comprehensive visualization toolkit for contaminant transport analysis.
    
    This class provides a complete suite of visualization tools for both
    numerical simulation results and machine learning model performance
    in contaminant transport studies. It supports static plots, animated
    visualizations, and comprehensive dashboards.
    
    Features:
        - Concentration field visualization (static and animated)
        - Risk level mapping and evolution
        - Machine learning model comparison dashboards
        - Confusion matrix analysis
        - Temporal snapshot generation
        - Video creation for concentration evolution
        
    Attributes:
        config (dict): Configuration parameters loaded from YAML file
        viz_config (dict): Visualization-specific configuration
        risk_colors (dict): Color mapping for risk levels (Low, Medium, High)
        risk_labels (dict): Label mapping for risk categories
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the contaminant visualization toolkit.
        
        This method sets up the visualization environment by loading configuration
        parameters, setting up matplotlib/seaborn styling, and defining color
        schemes for risk level visualization.
        
        Args:
            config_path (str, optional): Path to YAML configuration file.
                If None, defaults to '../../config/parameters.yaml'
                
        Note:
            Automatically configures seaborn styling and custom color palettes
            for consistent visualization appearance across all plots.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 
                                     '../../config/parameters.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.viz_config = self.config['visualization']
        
        # Configure style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure font sizes from config
        font_config = self.viz_config.get('font_size', {})
        self.font_sizes = {
            'title': font_config.get('title', 16),
            'label': font_config.get('label', 14),
            'tick': font_config.get('tick', 12),
            'legend': font_config.get('legend', 12),
            'annotation': font_config.get('annotation', 10)
        }
        
        # Apply global font settings
        plt.rcParams.update({
            'font.size': self.font_sizes['tick'],
            'axes.titlesize': self.font_sizes['title'],
            'axes.labelsize': self.font_sizes['label'],
            'xtick.labelsize': self.font_sizes['tick'],
            'ytick.labelsize': self.font_sizes['tick'],
            'legend.fontsize': self.font_sizes['legend'],
            'figure.titlesize': self.font_sizes['title'] + 2
        })
        
        # Custom colors for risk levels
        self.risk_colors = {
            0: '#2E8B57',  # Green for low risk
            1: '#FFD700',  # Yellow for medium risk
            2: '#DC143C'   # Red for high risk
        }
        
        self.risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    def plot_model_comparison(self, cv_scores: Dict[str, Dict], 
                            save_path: str = None) -> None:
        """
        Create a comparative bar chart of machine learning model performance.
        
        This method visualizes cross-validation accuracy scores for different
        machine learning models, displaying both mean accuracy and standard
        deviation as error bars. Values are annotated on each bar for clarity.
        
        Args:
            cv_scores (Dict[str, Dict]): Dictionary containing cross-validation
                results with structure: {model_name: {'mean': float, 'std': float}}
            save_path (str, optional): Path to save the figure. If None,
                the plot is only displayed.
                
        Note:
            The plot includes error bars representing standard deviation and
            rotated x-axis labels for better readability when model names are long.
        """
        # Prepare data
        models = list(cv_scores.keys())
        means = [cv_scores[model]['mean'] for model in models]
        stds = [cv_scores[model]['std'] for model in models]
        
        # Bar plot - Using configuration parameters
        fig, ax = plt.subplots(figsize=self.viz_config['figure_size'])
        
        bars = ax.bar(models, means, yerr=stds, capsize=5, 
                     color=sns.color_palette("husl", len(models)),
                     alpha=0.8, edgecolor='black')
        
        # Add values on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontweight='bold', 
                   fontsize=self.font_sizes['annotation'])
        
        ax.set_ylabel('Accuracy (Cross Validation)')
        ax.set_title('Machine Learning Models Comparison')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        plt.show()
    

    
    def plot_confusion_matrix_detailed(self, conf_matrix: np.ndarray,
                                      class_names: List[str] = None,
                                      save_path: str = None) -> None:
        """
        Create detailed confusion matrix visualization with absolute and normalized views.
        
        This method generates a comprehensive confusion matrix visualization
        showing both absolute counts and normalized percentages side by side.
        This dual view helps understand both the raw performance and the
        relative accuracy across different risk classes.
        
        Args:
            conf_matrix (np.ndarray): 2D confusion matrix array with shape (n_classes, n_classes)
            class_names (List[str], optional): List of class labels. 
                Defaults to ['Low', 'Medium', 'High'] for risk levels.
            save_path (str, optional): Path to save the figure. If None,
                the plot is only displayed.
                
        Note:
            The left plot shows absolute counts while the right plot shows
            row-normalized percentages. Both use the same color scheme for consistency.
        """
        if class_names is None:
            class_names = ['Low', 'Medium', 'High']
        
        # Normalize matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Using configuration parameters for figure size
        fig_width, fig_height = self.viz_config['figure_size']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width * 1.25, fig_height * 0.75))
        
        # Absolute matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Absolute Values)')
        ax1.set_xlabel('Prediction')
        ax1.set_ylabel('True Value')
        
        # Normalized matrix
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('True Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        plt.show()
        
    def create_simulation_video(self, concentration_history: np.ndarray,
                              x: np.ndarray, y: np.ndarray, 
                              time_points: np.ndarray,
                              scenario_name: str = "simulation",
                              save_path: str = None,
                              fps: int = None) -> None:
        """
        Generate animated video of contaminant concentration evolution over time.
        
        This method creates a time-lapse animation showing how contaminant
        concentration spreads and evolves throughout the simulation domain.
        The animation includes contour plots with color-coded concentration
        levels and contour lines for detailed visualization.
        
        Args:
            concentration_history (np.ndarray): 3D array with shape (time, y, x)
                containing concentration values at each time step
            x (np.ndarray): 1D array of x-coordinate values for the spatial grid
            y (np.ndarray): 1D array of y-coordinate values for the spatial grid
            time_points (np.ndarray): 1D array of time values corresponding to each frame
            scenario_name (str, optional): Name identifier for the simulation scenario.
                Defaults to "simulation".
            save_path (str, optional): Output file path for the video. If None,
                defaults to "data/results/{scenario_name}_animation.gif"
            fps (int, optional): Frames per second for the animation. Defaults to 10.
                
        Note:
            The method automatically adjusts figure dimensions for river-like
            geometries (20:1 aspect ratio) and uses consistent color scaling
            across all frames for accurate temporal comparison.
        """
        if save_path is None:
            save_path = f"data/results/{scenario_name}_animation.gif"
        
        # Get animation parameters from config
        anim_config = self.viz_config.get('animation', {})
        if fps is None:
            fps = anim_config.get('fps', 10)
        
        # Optimize frame count for faster generation
        max_frames = anim_config.get('max_frames', 100)
        frame_skip = anim_config.get('frame_skip', 1)
        
        # Reduce frames if necessary
        if len(time_points) > max_frames:
            indices = np.linspace(0, len(time_points)-1, max_frames, dtype=int)
            concentration_history = concentration_history[indices]
            time_points = time_points[indices]
        elif frame_skip > 1:
            indices = np.arange(0, len(time_points), frame_skip)
            concentration_history = concentration_history[indices]
            time_points = time_points[indices]
        
        # Create figure - Adjusted figsize for river geometry (200m x 10m = 20:1 ratio)
        # Using configuration parameters for figure proportions
        fig_width, fig_height = self.viz_config['figure_size']
        fig, ax = plt.subplots(figsize=(fig_width * 1.33, fig_height * 0.5))
        
        # Create mesh
        X, Y = np.meshgrid(x, y)
        
        # Configure color limits based on all data
        vmin = np.min(concentration_history)
        vmax = np.max(concentration_history)
        
        def animate(frame):
            ax.clear()
            
            # Check that frame is in range
            if frame >= len(concentration_history) or frame >= len(time_points):
                return []
            
            # Contour plot for this frame - Using configuration parameters
            contour = ax.contourf(X, Y, concentration_history[frame], 
                                levels=self.viz_config['contour_levels'], 
                                cmap=self.viz_config['colormap'], vmin=vmin, vmax=vmax)
            
            # Contour lines
            contour_lines = ax.contour(X, Y, concentration_history[frame], 
                                     levels=10, colors='white', alpha=0.6, linewidths=0.5)
            
            # Configuration
            ax.set_xlabel('Distance X (m)')
            ax.set_ylabel('Distance Y (m)')
            ax.set_title(f'{scenario_name} - Time: {time_points[frame]:.1f}s')
            ax.grid(True, alpha=0.3)
            # Removed axis equal for better river visualization (200m x 10m)
            
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(time_points), 
                           interval=1000//fps, blit=False, repeat=True)
        
        # Save animation with optimized DPI for videos
        video_dpi = anim_config.get('dpi_video', self.viz_config['dpi'])
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=video_dpi)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=video_dpi)
        
        plt.close(fig)
        
    def plot_all_model_metrics(self, model_results: Dict[str, Dict], 
                               save_path: str = None) -> None:
        """
        Visualize all metrics from all models in a complete dashboard.
        
        Args:
            model_results: Dictionary with results from all models
            save_path: Path to save the figure
        """
        # Prepare data
        models = list(model_results.keys())
        
        # Create figure with subplots - Using configuration parameters
        fig_width, fig_height = self.viz_config['figure_size']
        fig, axes = plt.subplots(2, 2, figsize=(fig_width * 1.25, fig_height * 1.5))
        fig.suptitle('Machine Learning Models Metrics Dashboard', 
                    fontsize=self.font_sizes['title'] + 2, fontweight='bold')
        
        # Colors for each model
        colors = sns.color_palette("husl", len(models))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        accuracy_scores = [model_results[model]['accuracy'] for model in models]
        
        bars1 = ax1.bar(models, accuracy_scores, color=colors, alpha=0.8)
        ax1.set_title('Accuracy on Test Set')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, acc in zip(bars1, accuracy_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=self.font_sizes['annotation'])
        
        # Plot 2: Classification metrics (heatmap)
        ax2 = axes[0, 1]
        metrics_data = []
        for model in models:
            metrics_data.append([
                model_results[model]['precision_weighted'],
                model_results[model]['recall_weighted'],
                model_results[model]['f1_weighted']
            ])
        
        metrics_df = pd.DataFrame(metrics_data, 
                                 index=models, 
                                 columns=['Precision', 'Recall', 'F1-Score'])
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'Metric Value'})
        ax2.set_title('Weighted Metrics on Test Set')
        
        # Plot 3: Macro vs weighted metrics comparison
        ax3 = axes[1, 0]
        x_pos = np.arange(len(models))
        width = 0.35
        
        # Bars for macro and weighted metrics
        macro_f1 = [model_results[model]['f1_macro'] for model in models]
        weighted_f1 = [model_results[model]['f1_weighted'] for model in models]
        
        bars1 = ax3.bar(x_pos - width/2, macro_f1, width, 
                       label='F1-Score Macro', alpha=0.8, color='skyblue')
        bars2 = ax3.bar(x_pos + width/2, weighted_f1, width,
                       label='F1-Score Weighted', alpha=0.8, color='lightcoral')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score Comparison: Macro vs Weighted')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1.1)
        
        # Add values on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=self.font_sizes['annotation'] - 2)
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=self.font_sizes['annotation'] - 2)
        
        # Plot 4: Metrics per class
        ax4 = axes[1, 1]
        classes = ['Low', 'Medium', 'High']
        x_pos = np.arange(len(classes))
        width = 0.2
        
        # Get per-class metrics for best model (highest accuracy)
        best_model = max(models, key=lambda x: model_results[x]['accuracy'])
        precision_per_class = model_results[best_model]['precision_per_class']
        recall_per_class = model_results[best_model]['recall_per_class']
        f1_per_class = model_results[best_model]['f1_per_class']
        
        bars1 = ax4.bar(x_pos - width, precision_per_class, width, 
                       label='Precision', alpha=0.8)
        bars2 = ax4.bar(x_pos, recall_per_class, width,
                       label='Recall', alpha=0.8)
        bars3 = ax4.bar(x_pos + width, f1_per_class, width,
                       label='F1-Score', alpha=0.8)
        
        ax4.set_xlabel('Risk Classes')
        ax4.set_ylabel('Metric Value')
        ax4.set_title(f'Metrics per Class - {best_model}')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(classes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        plt.show()
    
    def create_simulation_snapshots(self, concentration_history: np.ndarray,
                                  x: np.ndarray, y: np.ndarray,
                                  time_points: np.ndarray,
                                  scenario_name: str = "simulation",
                                  save_dir: str = None,
                                  num_snapshots: int = 4) -> List[str]:
        """
        Create multiple simulation images at different time points.
        
        Args:
            concentration_history: 3D array (time, y, x) with concentration evolution
            x: X coordinates
            y: Y coordinates
            time_points: Time points
            scenario_name: Scenario name
            save_dir: Directory to save images
            num_snapshots: Number of snapshots to generate
            
        Returns:
            List of saved file paths
        """
        if save_dir is None:
            save_dir = "data/results/snapshots"
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Select uniformly distributed time indices
        total_frames = len(time_points)
        if num_snapshots >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, num_snapshots, dtype=int)
        
        saved_files = []
        
        # Create mesh
        X, Y = np.meshgrid(x, y)
        
        # Configure color limits based on all data
        vmin = np.min(concentration_history)
        vmax = np.max(concentration_history)
        
        for i, frame_idx in enumerate(frame_indices):
            # Adjusted figsize for river geometry (200m x 10m = 20:1 ratio)
            # Using configuration parameters
            fig_width, fig_height = self.viz_config['figure_size']
            fig, ax = plt.subplots(figsize=(fig_width * 1.33, fig_height * 0.5))
            
            # Contour plot - Using configuration parameters
            contour = ax.contourf(X, Y, concentration_history[frame_idx], 
                                levels=self.viz_config['contour_levels'], 
                                cmap=self.viz_config['colormap'], vmin=vmin, vmax=vmax)
            
            # Contour lines
            contour_lines = ax.contour(X, Y, concentration_history[frame_idx], 
                                     levels=10, colors='white', alpha=0.6, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=self.font_sizes['annotation'] - 2, fmt='%.3f')
            
            # Color bar
            #cbar = plt.colorbar(contour, ax=ax)
            #cbar.set_label('Concentration (mg/L)', rotation=270, labelpad=20)
            
            # Configuration
            ax.set_xlabel('Distance X (m)')
            ax.set_ylabel('Distance Y (m)')
            ax.set_title(f'{scenario_name} - Time: {time_points[frame_idx]:.1f}s')
            ax.grid(True, alpha=0.3)
            # Removed axis equal for better river visualization (200m x 10m)
            
            plt.tight_layout()
            
            # Save image
            filename = f"{scenario_name}_t{time_points[frame_idx]:.1f}s_snapshot_{i+1}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            saved_files.append(filepath)

        
        return saved_files
    
    def create_concentration_snapshot(self, concentration: np.ndarray,
                                    x: np.ndarray, y: np.ndarray,
                                    time: float,
                                    scenario_name: str = "simulation",
                                    save_path: str = None) -> str:
        """
        Create a concentration image at a specific time.
        
        Args:
            concentration: 2D concentration array
            x: X coordinates
            y: Y coordinates
            time: Current time
            scenario_name: Scenario name
            save_path: Path to save the image
            
        Returns:
            Path of saved file
        """
        # Create mesh
        X, Y = np.meshgrid(x, y)
        
        # Adjusted figsize for river geometry (200m x 10m = 20:1 ratio)
        # Using configuration parameters
        fig_width, fig_height = self.viz_config['figure_size']
        fig, ax = plt.subplots(figsize=(fig_width * 1.33, fig_height * 0.5))
        
        # Contour plot - Using configuration parameters
        contour = ax.contourf(X, Y, concentration, 
                            levels=self.viz_config['contour_levels'], 
                            cmap=self.viz_config['colormap'])
        
        # Contour lines
        contour_lines = ax.contour(X, Y, concentration, levels=10, 
                                 colors='white', alpha=0.6, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=self.font_sizes['annotation'] - 2, fmt='%.3f')
        
        # Color bar
        #cbar = plt.colorbar(contour, ax=ax)
        #cbar.set_label('Concentration (mg/L)', rotation=270, labelpad=20)
        
        # Configuration
        ax.set_xlabel('Distance X (m)')
        ax.set_ylabel('Distance Y (m)')
        ax.set_title(f'{scenario_name} - Time: {time:.1f}s')
        ax.grid(True, alpha=0.3)
        # Removed axis equal for better river visualization (200m x 10m)
        
        plt.tight_layout()
        
        # Save image
        if save_path is None:
            save_path = f"data/results/{scenario_name}_t{time:.1f}s.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def create_risk_snapshots(self, concentration_history: np.ndarray,
                            x: np.ndarray, y: np.ndarray,
                            time_points: np.ndarray,
                            scenario_name: str = "risk_analysis",
                            save_dir: str = None,
                            num_snapshots: int = 4) -> List[str]:
        """
        Create multiple risk level images at different time points.
        
        Args:
            concentration_history: 3D array (time, y, x) with concentration evolution
            x: X coordinates
            y: Y coordinates
            time_points: Time points
            scenario_name: Scenario name
            save_dir: Directory to save images
            num_snapshots: Number of snapshots to generate
            
        Returns:
            List of saved file paths
        """
        if save_dir is None:
            save_dir = "data/results/risk_snapshots"
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get risk thresholds
        low_threshold = self.config['risk_thresholds']['low']
        medium_threshold = self.config['risk_thresholds']['medium']
        
        # Select uniformly distributed time indices
        total_frames = len(time_points)
        if num_snapshots >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, num_snapshots, dtype=int)
        
        saved_files = []
        
        # Create mesh
        X, Y = np.meshgrid(x, y)
        
        # Create custom colormap
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Yellow, Red
        cmap = LinearSegmentedColormap.from_list('risk', colors, N=3)
        
        for i, frame_idx in enumerate(frame_indices):
            # Adjusted figsize for river geometry (200m x 10m = 20:1 ratio)
            # Using configuration parameters
            fig_width, fig_height = self.viz_config['figure_size']
            fig, ax = plt.subplots(figsize=(fig_width * 1.33, fig_height * 0.5))
            
            # Calculate risk levels for this frame
            concentration = concentration_history[frame_idx]
            risk_levels = np.zeros_like(concentration, dtype=int)
            risk_levels[concentration > medium_threshold] = 2  # High risk
            risk_levels[(concentration > low_threshold) & (concentration <= medium_threshold)] = 1  # Medium risk
            
            # Risk plot
            im = ax.imshow(risk_levels, extent=[x.min(), x.max(), y.min(), y.max()],
                          origin='lower', cmap=cmap, vmin=0, vmax=2, aspect='auto')
            # Using aspect='auto' for proper river visualization (200m x 10m)
            
            # Custom color bar
            #cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
            #cbar.set_ticklabels(['Low', 'Medium', 'High'])
            #cbar.set_label('Risk Level', rotation=270, labelpad=20)
            
            # Configuration
            ax.set_xlabel('Distance X (m)')
            ax.set_ylabel('Distance Y (m)')
            ax.set_title(f'{scenario_name} - Ecological Risk - Time: {time_points[frame_idx]:.1f}s')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save image
            filename = f"{scenario_name}_risk_t{time_points[frame_idx]:.1f}s_snapshot_{i+1}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            saved_files.append(filepath)

        return saved_files
    
    def create_risk_evolution_video(self, concentration_history: np.ndarray,
                                  x: np.ndarray, y: np.ndarray,
                                  time_points: np.ndarray,
                                  scenario_name: str = "simulation",
                                  save_path: str = None,
                                  fps: int = None) -> None:
        """
        Create a video of risk zone evolution.
        
        Args:
            concentration_history: 3D array (time, y, x) with concentration evolution
            x: X coordinates
            y: Y coordinates
            time_points: Time points
            scenario_name: Scenario name
            save_path: Path to save the video
            fps: Frames per second
        """
        if save_path is None:
            save_path = f"data/results/{scenario_name}_risk_evolution.gif"
        
        # Get animation parameters from config
        anim_config = self.viz_config.get('animation', {})
        if fps is None:
            fps = anim_config.get('fps', 10)
        
        # Optimize frame count for faster generation
        max_frames = anim_config.get('max_frames', 100)
        frame_skip = anim_config.get('frame_skip', 1)
        
        # Reduce frames if necessary
        if len(time_points) > max_frames:
            indices = np.linspace(0, len(time_points)-1, max_frames, dtype=int)
            concentration_history = concentration_history[indices]
            time_points = time_points[indices]
        elif frame_skip > 1:
            indices = np.arange(0, len(time_points), frame_skip)
            concentration_history = concentration_history[indices]
            time_points = time_points[indices]
        
        # Get risk thresholds
        low_threshold = self.config['risk_thresholds']['low']
        medium_threshold = self.config['risk_thresholds']['medium']
        
        # Create figure
        # Using configuration parameters
        fig_width, fig_height = self.viz_config['figure_size']
        fig, ax = plt.subplots(figsize=(fig_width * 1.33, fig_height * 0.5))
        
        # Create mesh
        X, Y = np.meshgrid(x, y)
        
        # Create custom colormap
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Yellow, Red
        cmap = LinearSegmentedColormap.from_list('risk', colors, N=3)
        
        def animate(frame):
            ax.clear()
            
            # Check that frame is in range
            if frame >= len(concentration_history) or frame >= len(time_points):
                return []
            
            # Calculate risk levels for this frame
            concentration = concentration_history[frame]
            risk_levels = np.zeros_like(concentration, dtype=int)
            risk_levels[concentration > medium_threshold] = 2  # High risk
            risk_levels[(concentration > low_threshold) & (concentration <= medium_threshold)] = 1  # Medium risk
            
            # Plot
            im = ax.imshow(risk_levels, extent=[x.min(), x.max(), y.min(), y.max()],
                          origin='lower', cmap=cmap, vmin=0, vmax=2, aspect='auto')
            
            # Configuration
            ax.set_xlabel('Distance X (m)')
            ax.set_ylabel('Distance Y (m)')
            ax.set_title(f'{scenario_name} - Risk Evolution - Time: {time_points[frame]:.1f}s')
            ax.grid(True, alpha=0.3)
            
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(time_points), 
                           interval=1000//fps, blit=False, repeat=True)
        
        # Save animation with optimized DPI for videos
        video_dpi = anim_config.get('dpi_video', self.viz_config['dpi'])
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=video_dpi)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=video_dpi)
        
        plt.close(fig)