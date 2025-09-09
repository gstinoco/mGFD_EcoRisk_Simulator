"""
Hybrid Methodology for Ecological Risk Assessment in Water Bodies
Numerical Simulation + Machine Learning

Description:
    This script provides a comprehensive framework for ecological risk assessment in aquatic systems
    through the integration of numerical simulation and machine learning techniques. The methodology
    combines:
    - Advection-diffusion numerical modeling for contaminant transport simulation
    - Feature extraction from simulation results for machine learning applications
    - Risk classification using ensemble and traditional machine learning algorithms
    - Comprehensive visualization and reporting capabilities
    - Automated scenario generation and batch processing

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
import argparse
import os
import sys
import warnings

# Third-party libraries
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Configure warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Local modules
from ml_model.data_preprocessing import DataPreprocessor
from ml_model.risk_classifier import RiskClassifier
from numerical_model.advection_diffusion import AdvectionDiffusionModel
from visualization.visualization import ContaminantVisualizer

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary containing all simulation parameters
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def apply_scenario_parameters(config, scenario_params):
    """
    Apply scenario-specific parameters to the base configuration.
    
    Args:
        config (dict): Base configuration dictionary
        scenario_params (dict): Scenario-specific parameters to apply
        
    Returns:
        dict: Updated configuration with scenario parameters applied
    """
    # Mapping of scenario parameters to configuration paths
    param_mappings = {
        'velocity': {
            'u': ['physics', 'advection_velocity', 'u'],
            'v': ['physics', 'advection_velocity', 'v']
        },
        'diffusion_coefficient': ['physics', 'diffusion_coefficient'],
        'source_strength': ['source', 'strength'],
        'source_duration': ['source', 'duration'],
        'decay_rate': ['physics', 'decay_rate'],
        'boundary_type': ['boundary_conditions', 'type'],
        'total_time': ['domain', 'total_time']
    }
    
    for param_name, param_value in scenario_params.items():
        if param_name == 'source_location':
            # Handle source location (dict or list format)
            if isinstance(param_value, dict):
                config['source']['location']['x'] = param_value['x']
                config['source']['location']['y'] = param_value['y']
            else:
                # List format (backward compatibility)
                config['source']['location']['x'] = param_value[0]
                config['source']['location']['y'] = param_value[1]
        
        elif param_name == 'boundary_config':
            # Override entire boundary configuration
            config['boundary_conditions'] = param_value
        
        elif param_name == 'velocity':
            # Handle velocity components
            if 'u' in param_value:
                config['physics']['advection_velocity']['u'] = param_value['u']
            if 'v' in param_value:
                config['physics']['advection_velocity']['v'] = param_value['v']
        
        elif param_name in param_mappings:
            # Handle simple parameter mappings
            path = param_mappings[param_name]
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = param_value
    
    return config

def export_simulation_to_csv(scenario_path, export_concentration_history=False):
    """
    Export simulation data from NPY format to CSV format for analysis and collaboration.
    
    This function converts the binary NPY files to human-readable CSV format,
    making the data accessible for analysis in Excel, R, MATLAB, and other tools.
    
    Args:
        scenario_path (str): Path to the scenario directory containing NPY files
        export_concentration_history (bool): Whether to export full temporal history
                                           (can create very large files)
    
    Returns:
        dict: Paths to created CSV files
    """
    csv_files = {}
    
    try:
        # Load NPY data
        final_concentration = np.load(os.path.join(scenario_path, 'final_concentration.npy'))
        x_coords = np.load(os.path.join(scenario_path, 'x_coordinates.npy'))
        y_coords = np.load(os.path.join(scenario_path, 'y_coordinates.npy'))
        times = np.load(os.path.join(scenario_path, 'times.npy'))
        
        # Export final concentration as structured CSV
        final_data = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                final_data.append({
                    'x_coordinate': x,
                    'y_coordinate': y,
                    'concentration_mg_L': final_concentration[i, j],
                    'time_final': times[-1]
                })
        
        final_df = pd.DataFrame(final_data)
        final_csv_path = os.path.join(scenario_path, 'final_concentration.csv')
        final_df.to_csv(final_csv_path, index=False)
        csv_files['final_concentration'] = final_csv_path
        
        # Export coordinates separately
        coords_df = pd.DataFrame({
            'x_coordinates': x_coords,
            'y_coordinates': y_coords
        })
        coords_csv_path = os.path.join(scenario_path, 'coordinates.csv')
        coords_df.to_csv(coords_csv_path, index=False)
        csv_files['coordinates'] = coords_csv_path
        
        # Export times
        times_df = pd.DataFrame({'time_seconds': times})
        times_csv_path = os.path.join(scenario_path, 'times.csv')
        times_df.to_csv(times_csv_path, index=False)
        csv_files['times'] = times_csv_path
        
        # Optionally export concentration history (warning: can be very large)
        if export_concentration_history:
            try:
                concentration_history = np.load(os.path.join(scenario_path, 'concentration_history.npy'))
                history_data = []
                
                for t_idx, time_val in enumerate(times):
                    for i, x in enumerate(x_coords):
                        for j, y in enumerate(y_coords):
                            history_data.append({
                                'time_seconds': time_val,
                                'x_coordinate': x,
                                'y_coordinate': y,
                                'concentration_mg_L': concentration_history[t_idx, i, j]
                            })
                
                history_df = pd.DataFrame(history_data)
                history_csv_path = os.path.join(scenario_path, 'concentration_history.csv')
                history_df.to_csv(history_csv_path, index=False)
                csv_files['concentration_history'] = history_csv_path
                
            except Exception as e:
                print(f"Warning: Could not export concentration history: {e}")
        
        return csv_files
        
    except FileNotFoundError as e:
        print(f"Error: Required NPY files not found in {scenario_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return {}

def save_simulation_results(scenario_name, concentration_history, model, times, config, export_csv=False):
    """
    Save simulation results to disk.
    
    Args:
        scenario_name (str): Name of the scenario
        concentration_history (list): Time series of concentration fields
        model: AdvectionDiffusionModel instance
        times (np.ndarray): Time points array
        config (dict): Configuration parameters
        export_csv (bool): Whether to also export data in CSV format
    
    Returns:
        dict: Information about saved files
    """
    output_dir = f"data/simulations/{scenario_name or 'base'}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save simulation data
    files_to_save = {
        'final_concentration.npy': concentration_history[-1],
        'concentration_history.npy': np.array(concentration_history),
        'x_coordinates.npy': model.x,
        'y_coordinates.npy': model.y,
        'times.npy': times
    }
    
    saved_files = {'npy_files': []}
    for filename, data in files_to_save.items():
        file_path = os.path.join(output_dir, filename)
        np.save(file_path, data)
        saved_files['npy_files'].append(file_path)
    
    # Save parameters
    params_path = os.path.join(output_dir, 'parameters.yaml')
    with open(params_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    saved_files['parameters'] = params_path
    
    # Optionally export to CSV format
    if export_csv:
        print(f"Exporting {scenario_name} to CSV format...")
        include_history = config.get('output', {}).get('csv_include_history', False)
        csv_files = export_simulation_to_csv(output_dir, export_concentration_history=include_history)
        saved_files['csv_files'] = csv_files
        if csv_files:
            print(f"CSV files created: {list(csv_files.keys())}")
            if include_history:
                print("⚠️  Warning: concentration_history.csv file may be very large")
    
    return saved_files

def print_banner(title, subtitle=None, width=60):
    """
    Print a formatted banner with title and optional subtitle.
    
    Args:
        title (str): Main title text
        subtitle (str, optional): Subtitle text
        width (int): Width of the banner
    """
    print("\n" + "="*width)
    print(title)
    if subtitle:
        print(subtitle)
    print("="*width)

def print_model_performance_summary(all_models_results):
    """
    Print a comprehensive summary of model performance metrics.
    
    Args:
        all_models_results (dict): Dictionary containing results for all models
    """
    print_banner("COMPLETE MODEL PERFORMANCE SUMMARY", width=70)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in all_models_results.items():
        accuracy = metrics['accuracy']
        print(f"\n{model_name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.3f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.3f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.3f}")
        print(f"  Precision (Weighted): {metrics['precision_weighted']:.3f}")
        print(f"  Recall (Weighted): {metrics['recall_weighted']:.3f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.3f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print(f"\nBEST MODEL: {best_model}")
    print(f"   Accuracy: {best_accuracy:.3f}")
    print("="*70)

def run_single_simulation(config_path, scenario_name=None, save_results=True):
    """
    Execute a single contaminant transport simulation.
    
    This function runs an advection-diffusion simulation with specified parameters,
    optionally applying scenario-specific configurations.
    
    Args:
        config_path (str): Path to the configuration YAML file
        scenario_name (str, optional): Name of the scenario to run. If None, uses base configuration
        save_results (bool): Whether to save simulation results to disk
        
    Returns:
        dict: Dictionary containing simulation results including:
            - concentration_history: Time series of concentration fields
            - times: Time points array
            - final_concentration: Final concentration field
            - x, y: Spatial coordinates
            - model: AdvectionDiffusionModel instance
            - results: Complete simulation results
            - parameters: Configuration parameters used
    """
    print_banner(f"RUNNING SIMULATION: {scenario_name or 'BASE'}", width=50)
    
    config = load_config(config_path)
    
    # Apply scenario parameters if specified
    if scenario_name:
        scenario_params = next((s for s in config['scenarios'] if s['name'] == scenario_name), None)
        if scenario_params:
            config = apply_scenario_parameters(config, scenario_params)
    
    # Create and run model
    model = AdvectionDiffusionModel(config=config)
    results = model.run_simulation()
    
    concentration_history = results['concentration_history']
    times = results['time_points']

    # Save results if requested
    if save_results:
        export_csv = config.get('output', {}).get('export_csv', False)
        save_simulation_results(scenario_name, concentration_history, model, times, config, export_csv=export_csv)
    
    return {
        'concentration_history': concentration_history,
        'times': times,
        'final_concentration': concentration_history[-1],
        'x': results['x'],
        'y': results['y'],
        'model': model,
        'results': results,
        'parameters': config
    }

def run_multiple_scenarios(config_path, save_results=True):
    """
    Execute multiple simulation scenarios to generate a comprehensive dataset.
    
    This function runs all scenarios defined in the configuration file,
    creating a diverse dataset for machine learning training.
    
    Args:
        config_path (str): Path to the configuration YAML file
        save_results (bool): Whether to save simulation results to disk
        
    Returns:
        dict: Dictionary mapping scenario names to their simulation results
    """
    print_banner("GENERATING MULTIPLE SCENARIOS", width=50)
    
    config = load_config(config_path)
    scenarios = config['scenarios']
    
    simulation_results = {}
    
    for scenario in tqdm(scenarios, desc="Running scenarios"):
        scenario_name = scenario['name']
        results = run_single_simulation(config_path, scenario_name, save_results)
        simulation_results[scenario_name] = results
    
    print(f"\nGenerated {len(simulation_results)} scenarios")
    return simulation_results

def preprocess_data(config_path, simulation_results=None, save_results=True, use_fundamental_features=False):
    """
    Preprocess simulation data for machine learning.
    
    This function converts raw simulation results into a structured dataset
    suitable for training machine learning models, with options for different
    feature extraction strategies.
    
    Args:
        config_path (str): Path to the configuration YAML file
        simulation_results (dict, optional): Dictionary of simulation results. 
                                           If None, loads from saved files
        save_results (bool): Whether to save processed data to disk
        use_fundamental_features (bool): If True, uses only fundamental input parameters.
                                       If False, uses derived features including
                                       distance, travel time, and Peclet numbers
        
    Returns:
        dict: Dictionary containing processed data including:
            - X_train, X_test: Feature matrices for training and testing
            - y_train, y_test: Target vectors for training and testing
            - feature_names: List of feature names
            - scaler: Fitted StandardScaler object
            - combined_data: Complete preprocessed DataFrame
            - use_fundamental_features: Feature type flag
    """
    print_banner("PREPROCESSING DATA FOR ML", width=50)
    
    if use_fundamental_features:
        print("Using fundamental input parameter features")
    else:
        print("Using complete derived features")
    
    preprocessor = DataPreprocessor(config_path)
    
    if simulation_results is None:
        # Load data from files
        simulation_results = load_simulation_results()
    
    all_data = []
    
    for scenario_name, results in tqdm(simulation_results.items(), desc="Processing"):
        # Create data structure compatible with extract_features
        temp_data = {
            'concentration_history': [results['final_concentration']],
            'time_points': [0],  # Only final concentration available
            'coordinates': {
                'X': np.meshgrid(results['x'], results['y'])[0],
                'Y': np.meshgrid(results['x'], results['y'])[1]
            },
            'parameters': results.get('parameters', {})
        }
        
        scenario_df = preprocessor.process_simulation_data(temp_data, scenario_name, 
                                                         use_fundamental_features=use_fundamental_features)
        all_data.append(scenario_df)
    
    # Combine data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Prepare for ML
    X, y, feature_names = preprocessor.prepare_ml_dataset(combined_data, 
                                                         use_fundamental_features=use_fundamental_features)
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y)
    
    if save_results:
        processed_dir = 'data/processed'
        os.makedirs(processed_dir, exist_ok=True)
        
        # Add suffix to differentiate feature types
        suffix = "_fundamental" if use_fundamental_features else "_complete"
        
        np.save(os.path.join(processed_dir, f'X_train{suffix}.npy'), X_train)
        np.save(os.path.join(processed_dir, f'X_test{suffix}.npy'), X_test)
        np.save(os.path.join(processed_dir, f'y_train{suffix}.npy'), y_train)
        np.save(os.path.join(processed_dir, f'y_test{suffix}.npy'), y_test)
        
        with open(os.path.join(processed_dir, f'feature_names{suffix}.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
    
    print(f"Features used ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': preprocessor.scaler,
        'combined_data': combined_data,
        'use_fundamental_features': use_fundamental_features
    }

def train_models(config_path, data_dict=None):
    """
    Train machine learning models for ecological risk classification.
    
    This function performs comprehensive model training including cross-validation,
    hyperparameter optimization, and final model evaluation.
    
    Args:
        config_path (str): Path to the configuration YAML file
        data_dict (dict, optional): Preprocessed data dictionary. If None, loads from files
        
    Returns:
        dict: Dictionary containing training results including:
            - classifier: Trained RiskClassifier instance
            - cv_scores: Cross-validation scores for all models
            - test_results: Final model evaluation on test set
            - best_model: Best performing model
            - all_models_results: Comprehensive metrics for all models
            - metrics_report: DataFrame with detailed metrics report
    """
    print_banner("TRAINING ML MODELS", width=50)
    
    if data_dict is None:
        # Load processed data
        data_dict = load_processed_data()
    
    classifier = RiskClassifier(config_path)
    classifier.feature_names = data_dict['feature_names']
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Cross-validation evaluation
    cv_scores = classifier.evaluate_models(X_train, y_train)
    
    # Hyperparameter optimization
    best_model_name = max(cv_scores.keys(), key=lambda x: cv_scores[x]['mean'])
    print(f"\nOptimizing hyperparameters for best model: {best_model_name}")
    print(f"Original CV score: {cv_scores[best_model_name]['mean']:.4f} ± {cv_scores[best_model_name]['std']:.4f}")
    
    tuning_results = classifier.hyperparameter_tuning(X_train, y_train, best_model_name)
    
    # Display optimization results
    if tuning_results:
        print(f"Optimized CV score: {tuning_results['best_score']:.4f}")
        print(f"Best parameters: {tuning_results['best_params']}")
        improvement = tuning_results['best_score'] - cv_scores[best_model_name]['mean']
        print(f"Performance improvement: {improvement:.4f} ({improvement/cv_scores[best_model_name]['mean']*100:.2f}%)")
    
    # Train best model (now uses the optimized model)
    final_best_model = classifier.train_best_model(X_train, y_train, cv_scores)
    
    # Evaluate on test set
    test_results = classifier.evaluate_on_test(X_test, y_test)
    
    # Test set evaluation completed silently
    
    all_models_results = classifier.evaluate_all_models_on_test(X_train, y_train, X_test, y_test)
    
    # Generate metrics report
    results_dir = 'data/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine file names based on feature type
    use_fundamental_features = data_dict.get('use_fundamental_features', False)
    if use_fundamental_features:
        metrics_report_path = os.path.join(results_dir, 'all_models_metrics_report (fundamental features).csv')
        model_path = os.path.join(results_dir, 'risk_classifier_model (fundamental features).pkl')
    else:
        metrics_report_path = os.path.join(results_dir, 'all_models_metrics_report.csv')
        model_path = os.path.join(results_dir, 'risk_classifier_model.pkl')
    
    metrics_df = classifier.generate_metrics_report(all_models_results, save_path=metrics_report_path)
    
    # Save model
    classifier.save_model(model_path, test_results)
    
    return {
        'classifier': classifier,
        'cv_scores': cv_scores,
        'tuning_results': tuning_results,
        'test_results': test_results,
        'best_model': final_best_model,
        'all_models_results': all_models_results,
        'metrics_report': metrics_df
    }

def visualize_results(config_path, simulation_results=None, ml_results=None, use_fundamental_features=False):
    """
    Generate comprehensive visualizations of simulation and ML results.
    
    This function creates various plots and charts to visualize model performance,
    feature importance, and comparison metrics.
    
    Args:
        config_path (str): Path to the configuration YAML file
        simulation_results (dict, optional): Dictionary of simulation results
        ml_results (dict, optional): Dictionary of ML training results
        use_fundamental_features (bool, optional): If True, adds suffix to filenames for fundamental features
    """
    print_banner("GENERATING VISUALIZATIONS", width=50)
    
    viz = ContaminantVisualizer(config_path)
    
    viz_dir = 'data/visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Determine filename suffix based on feature type
    suffix = " (fundamental features)" if use_fundamental_features else ""
    
    if ml_results:
        # Generate model comparison plot only if CV data is available
        if ml_results['cv_scores'] and len(ml_results['cv_scores']) > 0:
            model_comparison_path = os.path.join(viz_dir, f'model_comparison{suffix}.png')
            viz.plot_model_comparison(ml_results['cv_scores'], save_path=model_comparison_path)
        
        confusion_matrix_path = os.path.join(viz_dir, f'confusion_matrix_detailed{suffix}.png')
        viz.plot_confusion_matrix_detailed(ml_results['test_results']['confusion_matrix'], 
                                         save_path=confusion_matrix_path)
        
        feature_importance_path = os.path.join(viz_dir, f'feature_importance{suffix}.png')
        feature_importance_df = ml_results['classifier'].plot_feature_importance(save_path=feature_importance_path)
        
        # Complete metrics dashboard and consolidated summary
        if 'all_models_results' in ml_results:
            all_metrics_dashboard_path = os.path.join(viz_dir, f'all_models_metrics_dashboard{suffix}.png')
            viz.plot_all_model_metrics(ml_results['all_models_results'], save_path=all_metrics_dashboard_path)
            
            # Consolidated performance summary
            print_model_performance_summary(ml_results['all_models_results'])
            
            print("\nTop 10 most important features:")
            print(feature_importance_df.nlargest(10, 'importance'))
        else:
            print("\nTop 10 most important features:")
            print(feature_importance_df.nlargest(10, 'importance'))

def load_all_models_metrics(use_fundamental_features=False):
    """
    Load metrics for all models from CSV file.
    
    Args:
        use_fundamental_features (bool): If True, loads metrics for fundamental features model.
                                       If False, loads metrics for complete features model.
    
    Returns:
        dict: Dictionary containing metrics for all models, or None if file not found
    """
    import pandas as pd
    
    # Determine the correct file path based on feature type
    if use_fundamental_features:
        csv_path = 'data/results/all_models_metrics_report (fundamental features).csv'
    else:
        csv_path = 'data/results/all_models_metrics_report.csv'
    
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} not found")
        return None
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Convert to JSON format for visualization
        metrics_dict = {}
        
        for _, row in df.iterrows():
            model_name = row['Model']
            metrics_dict[model_name] = {
                'accuracy': row['Accuracy'],
                'precision_macro': row['Precision (Macro)'],
                'recall_macro': row['Recall (Macro)'],
                'f1_macro': row['F1-Score (Macro)'],
                'precision_weighted': row['Precision (Weighted)'],
                'recall_weighted': row['Recall (Weighted)'],
                'f1_weighted': row['F1-Score (Weighted)'],
                'precision_per_class': [row['Precision Low'], row['Precision Medium'], row['Precision High']],
                'recall_per_class': [row['Recall Low'], row['Recall Medium'], row['Recall High']],
                'f1_per_class': [row['F1 Low'], row['F1 Medium'], row['F1 High']]
            }
        
        return metrics_dict
    except Exception as e:
        print(f"Error loading all models metrics: {e}")
        return None

def load_ml_results(use_fundamental_features=False):
    """
    Load machine learning results and test data.
    
    Args:
        use_fundamental_features (bool): If True, loads data with fundamental features (8 variables).
                                       If False, loads data with complete features (16 variables).
                                       Default: False.
    
    Returns:
        dict: Dictionary with ML results including:
            - classifier: Trained model
            - test_results: Test data results (confusion matrix, report, predictions)
            - cv_scores: Cross-validation scores
            - all_models_results: Metrics for all models (if available)
        
        None: If an error occurs during loading
    
    Note:
        This function automatically selects the correct data files based on
        the use_fundamental_features parameter to ensure compatibility with the trained model.
        - Fundamental features: source_x, source_y, velocity_u, velocity_v, 
          source_strength, x_position, y_position, time_normalized
        - Complete features: includes fundamental + 8 derived features
    """
    from src.ml_model.risk_classifier import RiskClassifier
    
    try:
        # Cargar modelo entrenado
        classifier = RiskClassifier()
        
        # Determine model path based on feature type
        if use_fundamental_features:
            model_path = 'data/results/risk_classifier_model (fundamental features).pkl'
        else:
            model_path = 'data/results/risk_classifier_model.pkl'
        
        classifier.load_model(model_path)
        
        # Determine suffix based on feature type
        # "_fundamental" for 8 features, "_complete" for 16 features
        suffix = "_fundamental" if use_fundamental_features else "_complete"
        
        # Load test data with appropriate suffix to ensure compatibility
        # with the trained model (avoids dimensionality errors)
        try:
            X_test = np.load(f'data/processed/X_test{suffix}.npy')
            y_test = np.load(f'data/processed/y_test{suffix}.npy')
        except FileNotFoundError:
            # Fallback to files without suffix (backward compatibility)
            print(f"Warning: Files with suffix {suffix} not found, using default files")
            X_test = np.load('data/processed/X_test.npy')
            y_test = np.load('data/processed/y_test.npy')
        
        # Make predictions
        y_pred, y_proba = classifier.predict_risk(X_test)
        
        # Calculate metrics
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Load all models metrics
        all_models_metrics = load_all_models_metrics(use_fundamental_features)
        
        results = {
            'classifier': classifier,
            'test_results': {
                'confusion_matrix': cm,
                'classification_report': report,
                'y_true': y_test,
                'y_pred': y_pred
            },
            'cv_scores': {}  # Placeholder for cross-validation scores
        }
        
        # Add all models metrics if available
        if all_models_metrics:
            results['all_models_results'] = all_models_metrics
        
        return results
    except Exception as e:
        print(f"Error loading ML results: {e}")
        return None

def export_existing_simulations_to_csv(export_concentration_history=False):
    """
    Export all existing simulation results from NPY to CSV format.
    
    This utility function converts all previously saved NPY simulation files
    to CSV format for analysis in external tools.
    
    Args:
        export_concentration_history (bool): Whether to export full temporal history
                                           (warning: creates very large files)
    
    Returns:
        dict: Summary of exported files by scenario
    """
    print_banner("EXPORTING SIMULATIONS TO CSV", width=50)
    
    simulations_dir = 'data/simulations'
    if not os.path.exists(simulations_dir):
        print(f"Error: Directory {simulations_dir} not found")
        return {}
    
    export_summary = {}
    scenarios = [d for d in os.listdir(simulations_dir) if os.path.isdir(os.path.join(simulations_dir, d))]
    
    for scenario_name in tqdm(scenarios, desc="Exporting scenarios"):
        scenario_path = os.path.join(simulations_dir, scenario_name)
        
        # Check if NPY files exist
        required_files = ['final_concentration.npy', 'x_coordinates.npy', 'y_coordinates.npy', 'times.npy']
        if all(os.path.exists(os.path.join(scenario_path, f)) for f in required_files):
            csv_files = export_simulation_to_csv(scenario_path, export_concentration_history)
            if csv_files:
                export_summary[scenario_name] = csv_files
                print(f"✓ Exported {scenario_name}: {len(csv_files)} CSV files")
            else:
                print(f"✗ Failed to export {scenario_name}")
        else:
            print(f"⚠ Skipping {scenario_name}: Missing NPY files")
    
    print(f"\nExport complete: {len(export_summary)} scenarios exported")
    return export_summary

def load_simulation_results():
    """
    Load simulation results from saved files in the data/simulations directory.
    
    This function scans the simulations directory and loads all available
    scenario results including concentration fields, coordinates, and parameters.
    
    Returns:
        dict: Dictionary mapping scenario names to their simulation results.
              Each result contains:
              - concentration_history: List of concentration fields over time
              - times: Array of time points
              - final_concentration: Final concentration field
              - x, y: Spatial coordinate arrays
              - parameters: Configuration parameters used
              
    Raises:
        FileNotFoundError: If the simulations directory doesn't exist
        ValueError: If no valid simulation files are found
    """
    simulation_results = {}
    simulations_dir = 'data/simulations'
    
    if not os.path.exists(simulations_dir):
        raise FileNotFoundError(f"Directory {simulations_dir} not found")
    
    for scenario_name in os.listdir(simulations_dir):
        scenario_path = os.path.join(simulations_dir, scenario_name)
        if os.path.isdir(scenario_path):
            try:
                concentration = np.load(os.path.join(scenario_path, 'final_concentration.npy'))
                x = np.load(os.path.join(scenario_path, 'x_coordinates.npy'))
                y = np.load(os.path.join(scenario_path, 'y_coordinates.npy'))
                
                # Load parameters if they exist
                params_file = os.path.join(scenario_path, 'parameters.yaml')
                parameters = {}
                if os.path.exists(params_file):
                    with open(params_file, 'r') as f:
                        parameters = yaml.safe_load(f)
                
                simulation_results[scenario_name] = {
                    'final_concentration': concentration,
                    'x': x,
                    'y': y,
                    'parameters': parameters
                }
            except FileNotFoundError:
                print(f"Warning: Incomplete files for scenario {scenario_name}")
    
    return simulation_results

def load_processed_data(use_fundamental_features=False):
    """
    Load preprocessed machine learning data from saved files.
    
    This function loads the training and test datasets that have been
    preprocessed and saved to disk, including feature matrices, target
    vectors, and feature names.
    
    Args:
        use_fundamental_features (bool, optional): If True, loads data with
            fundamental input parameters only (8 features). If False, loads
            data with complete derived features (16 features). Defaults to False.
    
    Returns:
        dict: Dictionary containing the loaded data with keys:
            - X_train: Training feature matrix
            - X_test: Test feature matrix  
            - y_train: Training target vector
            - y_test: Test target vector
            - feature_names: List of feature names
            - use_fundamental_features: Boolean flag indicating feature type
            
    Raises:
        FileNotFoundError: If the processed data directory or files don't exist
    """
    processed_dir = 'data/processed'
    
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Directory {processed_dir} not found")
    
    # Determine suffix based on feature type
    suffix = "_fundamental" if use_fundamental_features else "_complete"
    
    # Try to load files with suffix, if they don't exist use files without suffix
    try:
        X_train = np.load(os.path.join(processed_dir, f'X_train{suffix}.npy'))
        X_test = np.load(os.path.join(processed_dir, f'X_test{suffix}.npy'))
        y_train = np.load(os.path.join(processed_dir, f'y_train{suffix}.npy'))
        y_test = np.load(os.path.join(processed_dir, f'y_test{suffix}.npy'))
        
        feature_names_path = os.path.join(processed_dir, f'feature_names{suffix}.txt')
    except FileNotFoundError:
        # Fallback to files without suffix
        X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        
        feature_names_path = os.path.join(processed_dir, 'feature_names.txt')
    
    feature_names = []
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'use_fundamental_features': use_fundamental_features
    }

def create_simulation_videos(config_path, fps=None):
    """
    Create temporal evolution videos for all simulation scenarios.
    
    This function generates animated GIF videos showing the temporal evolution
    of contaminant concentration and ecological risk levels for all available
    simulation scenarios. Videos are saved in the data/videos directory.
    
    Args:
        config_path (str): Path to the configuration YAML file containing
                          visualization parameters and risk thresholds.
        fps (int, optional): Frames per second for the generated videos. 
                           If None, uses value from config file (default: 15).
                          
    Note:
        This function requires simulation results to be available in the
        data/simulations directory. Videos are created for both concentration
        fields and risk level classifications.
        
    Raises:
        FileNotFoundError: If simulation results are not found
        RuntimeError: If video creation fails due to missing dependencies
    """
    print_banner("GENERATING SIMULATION VIDEOS", width=50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get animation parameters from config
    if fps is None:
        fps = config.get('visualization', {}).get('animation', {}).get('fps', 15)
    
    # Initialize visualizer
    from src.visualization.visualization import ContaminantVisualizer
    visualizer = ContaminantVisualizer(config_path)
    
    # Load simulation results
    simulation_results = load_simulation_results() if os.path.exists('data/simulations') else None
    
    if not simulation_results:
        print("No simulation results found. Run simulations first.")
        return
    
    videos_dir = 'data/videos'
    os.makedirs(videos_dir, exist_ok=True)
    
    for scenario_name, results in simulation_results.items():
        print(f"\nGenerating video for scenario: {scenario_name}")
        
        scenario_path = f'data/simulations/{scenario_name}'
        concentration_history_file = os.path.join(scenario_path, 'concentration_history.npy')
        time_points_file = os.path.join(scenario_path, 'times.npy')
        
        if os.path.exists(concentration_history_file) and os.path.exists(time_points_file):
            concentration_history = np.load(concentration_history_file)
            time_points = np.load(time_points_file)
            
            # Create concentration video
            video_path = os.path.join(videos_dir, f'{scenario_name}_concentration.gif')
            visualizer.create_simulation_video(
                concentration_history=concentration_history,
                x=results['x'],
                y=results['y'],
                time_points=time_points,
                scenario_name=scenario_name,
                save_path=video_path,
                fps=fps
            )
            print(f"Video saved: {video_path}")
            
            # Create risk evolution video
            risk_video_path = os.path.join(videos_dir, f'{scenario_name}_risk.gif')
            visualizer.create_risk_evolution_video(
                concentration_history=concentration_history,
                x=results['x'],
                y=results['y'],
                time_points=time_points,
                scenario_name=scenario_name,
                save_path=risk_video_path,
                fps=fps
            )
            print(f"Risk video saved: {risk_video_path}")
        else:
            print(f"Warning: No temporal data found for {scenario_name}")
    
    print(f"\nVideos generated in: {videos_dir}")

def create_simulation_snapshots(config_path, num_snapshots=4):
    """
    Create static snapshot images at different time points for all simulations.
    
    This function generates static images showing contaminant concentration
    and risk level distributions at specific time points during the simulation.
    Images are saved in the data/snapshots directory organized by scenario.
    
    Args:
        config_path (str): Path to the configuration YAML file containing
                          visualization parameters, time points, and risk thresholds.
        num_snapshots (int): Number of snapshots to generate at evenly spaced
                           time intervals. Default is 4.
                          
    Note:
        The function creates snapshots at predefined time intervals based on
        the simulation duration. Each scenario gets its own subdirectory with
        concentration and risk level images.
        
    Raises:
        FileNotFoundError: If simulation results are not found
        ValueError: If invalid time points are specified
    """
    print_banner("GENERATING SIMULATION SNAPSHOTS", width=50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize visualizer
    from src.visualization.visualization import ContaminantVisualizer
    visualizer = ContaminantVisualizer(config_path)
    
    # Load simulation results
    simulation_results = load_simulation_results() if os.path.exists('data/simulations') else None
    
    if not simulation_results:
        print("No simulation results found. Run simulations first.")
        return
    
    snapshots_dir = 'data/snapshots'
    os.makedirs(snapshots_dir, exist_ok=True)
    
    for scenario_name, results in simulation_results.items():
        print(f"\nGenerating snapshots for scenario: {scenario_name}")
        
        scenario_path = f'data/simulations/{scenario_name}'
        concentration_history_file = os.path.join(scenario_path, 'concentration_history.npy')
        time_points_file = os.path.join(scenario_path, 'times.npy')
        
        if os.path.exists(concentration_history_file) and os.path.exists(time_points_file):
            concentration_history = np.load(concentration_history_file)
            time_points = np.load(time_points_file)
            
            scenario_snapshots_dir = os.path.join(snapshots_dir, scenario_name)
            os.makedirs(scenario_snapshots_dir, exist_ok=True)
            
            # Generate concentration snapshots
            snapshot_paths = visualizer.create_simulation_snapshots(
                concentration_history=concentration_history,
                x=results['x'],
                y=results['y'],
                time_points=time_points,
                scenario_name=scenario_name,
                save_dir=scenario_snapshots_dir,
                num_snapshots=num_snapshots
            )
            
            # Generate risk snapshots
            risk_snapshot_paths = visualizer.create_risk_snapshots(
                concentration_history=concentration_history,
                x=results['x'],
                y=results['y'],
                time_points=time_points,
                scenario_name=scenario_name,
                save_dir=scenario_snapshots_dir,
                num_snapshots=num_snapshots
            )
            
            print(f"Concentration snapshots generated:")
            for path in snapshot_paths:
                print(f"  - {path}")
            print(f"Risk snapshots generated:")
            for path in risk_snapshot_paths:
                print(f"  - {path}")
        else:
            print(f"Warning: No temporal data found for {scenario_name}")
    
    print(f"\nSnapshots generated in: {snapshots_dir}")

def run_complete_workflow(config_path, use_fundamental_features=False):
    """Execute the complete workflow."""
    print_banner("WATER BODY CONTAMINANT ANALYSIS", width=50)
    print("Hybrid Methodology: Numerical Simulation + Machine Learning")
    if use_fundamental_features:
        print("Mode: Fundamental Features")
    else:
        print("Mode: Complete Derived Features")
    print("="*60)
    
    # 1. Generate simulations
    simulation_results = run_multiple_scenarios(config_path)

    # 2. Generate temporal evolution videos
    create_simulation_videos(config_path)
    
    # 3. Generate snapshots at different time points
    create_simulation_snapshots(config_path, num_snapshots=4)
    
    # 4. Preprocess data
    data_dict = preprocess_data(config_path, simulation_results, 
                               use_fundamental_features=use_fundamental_features)
    
    # 5. Train models
    ml_results = train_models(config_path, data_dict)
    
    # 6. Generate visualizations
    visualize_results(config_path, simulation_results, ml_results, use_fundamental_features=use_fundamental_features)
    
    print_banner("COMPLETE WORKFLOW EXECUTED SUCCESSFULLY", width=50)
    
    return {
        'simulation_results': simulation_results,
        'data_dict': data_dict,
        'ml_results': ml_results
    }

def main():
    """
    Main entry point for the contaminant analysis application.
    
    This function provides an interactive command-line interface for users
    to select and execute different components of the analysis workflow:
    
    1. Run complete workflow (simulation + ML + visualization)
    2. Run only simulations
    3. Preprocess data for ML
    4. Train ML models
    5. Create visualization videos
    6. Create snapshot images
    7. Compare feature approaches
    
    The function presents a menu-driven interface and executes the selected
    option using the configuration file 'parameters.yaml'.
    
    Note:
        Requires 'parameters.yaml' configuration file to be present in the
        current directory with all necessary parameters defined.
        
    Raises:
        FileNotFoundError: If parameters.yaml is not found
        KeyboardInterrupt: If user interrupts the execution
    """
    parser = argparse.ArgumentParser(
        description="Water body contaminant analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py --complete                                 # Execute complete workflow
  python main.py --complete --fundamental-features          # Complete workflow with fundamental features
  python main.py --simulate --scenario base                 # Simulate specific scenario
  python main.py --simulate --all-scenarios                 # Simulate all scenarios
  python main.py --preprocess                               # Only preprocess data (complete features)
  python main.py --preprocess --fundamental-features        # Preprocess with fundamental features

  python main.py --train                                    # Only train models
  python main.py --train --fundamental-features             # Train with fundamental features
  python main.py --visualize                                # Only generate visualizations (complete features)
  python main.py --visualize --fundamental-features         # Generate visualizations (fundamental features)
  python main.py --create-videos                            # Generate temporal evolution videos
  python main.py --create-snapshots                         # Generate images at different time points
  python main.py --create-snapshots --snapshots-count 6     # Generate 6 snapshots
        """
    )
    
    parser.add_argument('--config', '-c', 
                       default='config/parameters.yaml',
                       help='Path to configuration file (default: config/parameters.yaml)')
    
    parser.add_argument('--complete', 
                       action='store_true',
                       help='Execute complete workflow')
    
    parser.add_argument('--simulate', 
                       action='store_true',
                       help='Execute numerical simulations')
    
    parser.add_argument('--scenario', 
                       help='Name of specific scenario to simulate')
    
    parser.add_argument('--all-scenarios', 
                       action='store_true',
                       help='Simulate all defined scenarios')
    
    parser.add_argument('--preprocess', 
                       action='store_true',
                       help='Preprocess data for machine learning')
    
    parser.add_argument('--fundamental-features', 
                       action='store_true',
                       help='Use only fundamental features (source location, velocities, intensity)')
    
    parser.add_argument('--train', 
                       action='store_true',
                       help='Train machine learning models')
    
    parser.add_argument('--visualize', 
                       action='store_true',
                       help='Generate visualizations (use --fundamental-features if model was trained with fundamental features)')
    
    parser.add_argument('--create-videos', 
                       action='store_true',
                       help='Generate temporal evolution videos')
    
    parser.add_argument('--create-snapshots', 
                       action='store_true',
                       help='Generate images at different time points')
    
    parser.add_argument('--snapshots-count', 
                       type=int, default=4,
                       help='Number of snapshots to generate (default: 4)')
    
    parser.add_argument('--video-fps', 
                       type=int, default=None,
                       help='Frames per second for videos (default: from config file, 15 fps)')
    
    parser.add_argument('--no-save', 
                       action='store_true',
                       help='Do not save results (for testing only)')
    
    args = parser.parse_args()
    
    # Verify that the configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        return 1
    
    save_results = not args.no_save
    
    try:
        if args.complete:
            run_complete_workflow(args.config, use_fundamental_features=args.fundamental_features)
        
        elif args.simulate:
            if args.all_scenarios:
                run_multiple_scenarios(args.config, save_results)
            elif args.scenario:
                run_single_simulation(args.config, args.scenario, save_results)
            else:
                run_single_simulation(args.config, save_results=save_results)
        
        elif args.preprocess:
            preprocess_data(args.config, save_results=save_results, 
                          use_fundamental_features=args.fundamental_features)
        
        elif args.train:
            data_dict = load_processed_data(use_fundamental_features=args.fundamental_features)
            train_models(args.config, data_dict)
        
        elif args.visualize:
            simulation_results = load_simulation_results() if os.path.exists('data/simulations') else None
            # Load ML results with correct feature type to avoid dimensionality errors
            # Determine the correct model path to check
            if args.fundamental_features:
                model_path_to_check = 'data/results/risk_classifier_model (fundamental features).pkl'
            else:
                model_path_to_check = 'data/results/risk_classifier_model.pkl'
            
            ml_results = load_ml_results(use_fundamental_features=args.fundamental_features) if os.path.exists(model_path_to_check) else None
            visualize_results(args.config, simulation_results, ml_results, use_fundamental_features=args.fundamental_features)
        
        elif args.create_videos:
            create_simulation_videos(args.config, args.video_fps)
        
        elif args.create_snapshots:
            create_simulation_snapshots(args.config, args.snapshots_count)
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())