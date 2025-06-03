import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz


# Set up the app
st.set_page_config(page_title="Interactive Decision Tree Workshop", layout="wide")
st.title("🌳 Interactive Decision Tree Workshop")
st.write("Create your own machine learning experiment! Define features, collect data, and watch the algorithm learn.")

# Sidebar for configuration
st.sidebar.header("🔧 Configure Your Experiment")

# Initialize session state
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = ""
if 'target_options' not in st.session_state:
    st.session_state.target_options = []
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'experiment_configured' not in st.session_state:
    st.session_state.experiment_configured = False

# Feature configuration section
with st.sidebar.expander("📊 Define Your Features", expanded=not st.session_state.experiment_configured):
    st.write("**Features** are the characteristics you'll use to make predictions (like age, preferences, etc.)")
    
    # Add new feature
    with st.form("add_feature"):
        feature_name = st.text_input("Feature name (e.g., 'Age', 'Favourite Colour')")
        feature_type = st.selectbox("Feature type", ["Categorical", "Numerical"])
        
        if feature_type == "Categorical":
            options_text = st.text_input("Options (comma-separated)", placeholder="Red, Blue, Green")
        else:
            col1, col2 = st.columns(2)
            min_val = col1.number_input("Min value", value=0)
            max_val = col2.number_input("Max value", value=10)
        
        if st.form_submit_button("Add Feature"):
            if feature_name:
                feature_config = {
                    'name': feature_name,
                    'type': feature_type
                }
                if feature_type == "Categorical":
                    feature_config['options'] = [opt.strip() for opt in options_text.split(',') if opt.strip()]
                else:
                    feature_config['min'] = min_val
                    feature_config['max'] = max_val
                
                st.session_state.features.append(feature_config)
                st.success(f"Added feature: {feature_name}")

# Display current features
if st.session_state.features:
    st.sidebar.write("**Current Features:**")
    for i, feature in enumerate(st.session_state.features):
        col1, col2 = st.sidebar.columns([3, 1])
        if feature['type'] == 'Categorical':
            col1.write(f"• {feature['name']} ({', '.join(feature['options'])})")
        else:
            col1.write(f"• {feature['name']} ({feature['min']}-{feature['max']})")
        if col2.button("🗑️", key=f"delete_{i}"):
            st.session_state.features.pop(i)
            st.rerun()

# Target variable configuration
with st.sidebar.expander("🎯 Define What You're Predicting", expanded=not st.session_state.experiment_configured):
    st.write("**Target variable** is what you want to predict (like personality type, preferences, etc.)")
    
    with st.form("target_config"):
        target_name = st.text_input("What are you trying to predict?", placeholder="Personality Type")
        target_options_text = st.text_input("Possible outcomes (comma-separated)", placeholder="Introvert, Extrovert")
        
        if st.form_submit_button("Set Target"):
            if target_name and target_options_text:
                st.session_state.target_variable = target_name
                st.session_state.target_options = [opt.strip() for opt in target_options_text.split(',') if opt.strip()]
                st.success(f"Target set: {target_name}")

# Experiment ready check
experiment_ready = (len(st.session_state.features) > 0 and 
                   st.session_state.target_variable and 
                   len(st.session_state.target_options) > 1)

if experiment_ready and not st.session_state.experiment_configured:
    if st.sidebar.button("🚀 Start Experiment", type="primary"):
        st.session_state.experiment_configured = True
        # Initialize data structure
        columns = [f['name'] for f in st.session_state.features] + [st.session_state.target_variable]
        st.session_state.data = pd.DataFrame(columns=columns)
        st.rerun()

# Reset experiment
if st.sidebar.button("🔄 Reset Experiment"):
    for key in ['features', 'target_variable', 'target_options', 'data', 'experiment_configured']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Main content
if not experiment_ready:
    st.info("👈 Configure your experiment in the sidebar to get started!")
    
    # Show some example configurations
    st.subheader("💡 Example Experiments")
    
    examples = [
        {
            "title": "Music Preference Predictor",
            "features": ["Age (18-80)", "Favourite Genre (Rock, Pop, Classical)", "Morning/Night Person (Morning, Night)"],
            "target": "Spotify vs Apple Music"
        },
        {
            "title": "Study Method Predictor", 
            "features": ["Hours of Sleep (4-12)", "Coffee Intake (Low, Medium, High)", "Subject (Maths, Science, English)"],
            "target": "Preferred Study Time (Morning, Afternoon, Evening)"
        },
        {
            "title": "Weekend Activity Predictor",
            "features": ["Weather Preference (Sunny, Rainy)", "Social Level (1-10)", "Budget (Low, Medium, High)"],
            "target": "Activity Type (Indoor, Outdoor, Social, Solo)"
        }
    ]
    
    for example in examples:
        with st.expander(f"📋 {example['title']}"):
            st.write(f"**Features:** {', '.join(example['features'])}")
            st.write(f"**Predicting:** {example['target']}")

elif st.session_state.experiment_configured:
    # Data collection section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Collect Training Data")
        
        with st.form("data_collection"):
            st.write(f"Add data to predict: **{st.session_state.target_variable}**")
            
            form_data = {}
            
            # Create input fields for each feature
            for feature in st.session_state.features:
                if feature['type'] == 'Categorical':
                    form_data[feature['name']] = st.selectbox(
                        feature['name'], 
                        feature['options'],
                        key=f"input_{feature['name']}"
                    )
                else:
                    form_data[feature['name']] = st.slider(
                        feature['name'],
                        min_value=int(feature['min']),
                        max_value=int(feature['max']),
                        value=int((feature['min'] + feature['max']) / 2),
                        key=f"input_{feature['name']}"
                    )
            
            # Target variable input
            form_data[st.session_state.target_variable] = st.selectbox(
                f"Actual {st.session_state.target_variable}",
                st.session_state.target_options
            )
            
            if st.form_submit_button("➕ Add Data Point", type="primary"):
                # Convert categorical to numerical for the model
                model_data = {}
                display_data = {}
                
                for feature in st.session_state.features:
                    if feature['type'] == 'Categorical':
                        # Store display version
                        display_data[feature['name']] = form_data[feature['name']]
                        # Store numerical version for model
                        model_data[f"{feature['name']}_encoded"] = feature['options'].index(form_data[feature['name']])
                    else:
                        display_data[feature['name']] = form_data[feature['name']]
                        model_data[f"{feature['name']}_encoded"] = form_data[feature['name']]
                
                # Handle target variable
                display_data[st.session_state.target_variable] = form_data[st.session_state.target_variable]
                model_data[f"{st.session_state.target_variable}_encoded"] = st.session_state.target_options.index(form_data[st.session_state.target_variable])
                
                # Add to dataframe
                new_row = pd.DataFrame([{**display_data, **model_data}])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.success("✅ Data point added!")
    
    with col2:
        st.subheader("📊 Current Dataset")
        
        if not st.session_state.data.empty:
            # Show display version (human-readable)
            display_cols = [f['name'] for f in st.session_state.features] + [st.session_state.target_variable]
            display_df = st.session_state.data[display_cols]
            
            st.dataframe(display_df, use_container_width=True)
            st.write(f"**Total data points:** {len(display_df)}")
            
            # Show data distribution
            if len(display_df) > 0:
                target_counts = display_df[st.session_state.target_variable].value_counts()
                st.write(f"**Distribution of {st.session_state.target_variable}:**")
                st.bar_chart(target_counts)
        else:
            st.info("No data collected yet. Add some data points to train the model!")
    
    # Model training and visualization
    min_samples = max(2, len(st.session_state.target_options))
    
    if len(st.session_state.data) >= min_samples:
        st.subheader("🌳 Decision Tree Model")
        
        # Prepare data for model
        feature_cols = [f"{f['name']}_encoded" for f in st.session_state.features]
        target_col = f"{st.session_state.target_variable}_encoded"
        
        X = st.session_state.data[feature_cols]
        y = st.session_state.data[target_col]
        
        # Model parameters
        col1, col2, col3 = st.columns(3)
        max_depth = col1.slider("Max Tree Depth", 1, 10, 3)
        criterion = col2.selectbox("Split Criterion", ["entropy", "gini"])
        min_samples_split = col3.slider("Min Samples to Split", 2, 10, 2)
        
        # Train model
        clf = tree.DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=42
        )
        clf.fit(X, y)
        
        # Visualize tree
        try:
            dot_data = export_graphviz(
                clf,
                out_file=None,
                feature_names=[f['name'] for f in st.session_state.features],
                class_names=st.session_state.target_options,
                filled=True,
                rounded=True,
                special_characters=True
            )
            st.graphviz_chart(dot_data)
        except Exception as e:
            st.error(f"Could not display tree visualization: {str(e)}")
        
        # Model insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': [f['name'] for f in st.session_state.features],
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.write("**Which features matter most?**")
            st.bar_chart(importance_df.set_index('Feature'))
        
        with col2:
            st.subheader("🔮 Make Predictions")
            
            with st.form("prediction"):
                pred_data = {}
                
                for feature in st.session_state.features:
                    if feature['type'] == 'Categorical':
                        pred_data[feature['name']] = st.selectbox(
                            feature['name'],
                            feature['options'],
                            key=f"pred_{feature['name']}"
                        )
                    else:
                        pred_data[feature['name']] = st.slider(
                            feature['name'],
                            min_value=int(feature['min']),
                            max_value=int(feature['max']),
                            value=int((feature['min'] + feature['max']) / 2),
                            key=f"pred_{feature['name']}"
                        )
                
                if st.form_submit_button("🔮 Predict", type="primary"):
                    # Convert to model format
                    pred_input = []
                    for feature in st.session_state.features:
                        if feature['type'] == 'Categorical':
                            pred_input.append(feature['options'].index(pred_data[feature['name']]))
                        else:
                            pred_input.append(pred_data[feature['name']])
                    
                    pred_input = np.array([pred_input])
                    
                    # Make prediction
                    prediction = int(clf.predict(pred_input)[0])
                    prediction_prob = clf.predict_proba(pred_input)[0]
                    
                    predicted_class = st.session_state.target_options[prediction]
                    confidence = prediction_prob[prediction]
                    
                    # Display result with confidence
                    if confidence > 0.8:
                        confidence_msg = "Very confident! 🎯"
                    elif confidence > 0.6:
                        confidence_msg = "Fairly confident 👍"
                    else:
                        confidence_msg = "Not very sure 🤔"
                    
                    st.success(f"**Prediction:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.1%} - {confidence_msg}")
                    
                    # Show probability breakdown
                    prob_df = pd.DataFrame({
                        'Outcome': st.session_state.target_options,
                        'Probability': prediction_prob
                    })
                    st.write("**Prediction Probabilities:**")
                    st.bar_chart(prob_df.set_index('Outcome'))
    
    else:
        st.warning(f"⏳ Need at least {min_samples} data points to train the model. You have {len(st.session_state.data)}.")
    
    # Interactive features section
    if len(st.session_state.data) >= min_samples:
        st.subheader("🎮 Interactive Exploration")
        
        tabs = st.tabs(["🔄 What-If Analysis", "📊 data Exploration", "🧪 Experiment"])
        
        with tabs[0]:
            st.write("See how changing one feature affects predictions:")
            
            if st.session_state.features:
                selected_feature = st.selectbox("Choose a feature to vary:", 
                                               [f['name'] for f in st.session_state.features])
                
                # Find the selected feature config
                feature_config = next(f for f in st.session_state.features if f['name'] == selected_feature)
                
                # Create base prediction input
                base_input = {}
                for feature in st.session_state.features:
                    if feature['name'] != selected_feature:
                        if feature['type'] == 'Categorical':
                            base_input[feature['name']] = st.selectbox(
                                f"Base {feature['name']}", 
                                feature['options'],
                                key=f"base_{feature['name']}"
                            )
                        else:
                            base_input[feature['name']] = st.slider(
                                f"Base {feature['name']}",
                                min_value=int(feature['min']),
                                max_value=int(feature['max']),
                                value=int((feature['min'] + feature['max']) / 2),
                                key=f"base_{feature['name']}"
                            )
                
                # Generate predictions across the selected feature
                predictions_list = []
                
                if feature_config['type'] == 'Categorical':
                    for option in feature_config['options']:
                        test_input = base_input.copy()
                        test_input[selected_feature] = option
                        
                        # Convert to model format
                        pred_input = []
                        for feature in st.session_state.features:
                            if feature['type'] == 'Categorical':
                                pred_input.append(feature['options'].index(test_input[feature['name']]))
                            else:
                                pred_input.append(test_input[feature['name']])
                        
                        pred_input = np.array([pred_input])
                        prediction = int(clf.predict(pred_input)[0])
                        pred_prob = clf.predict_proba(pred_input)[0]
                        
                        predictions_list.append({
                            selected_feature: option,
                            'Predicted_Class': st.session_state.target_options[prediction],
                            'Confidence': pred_prob[prediction]
                        })
                
                if predictions_list:
                    pred_df = pd.DataFrame(predictions_list)
                    st.write(f"**How {selected_feature} affects predictions:**")
                    
                    # Create a simple chart showing predictions
                    chart_data = pred_df.pivot_table(
                        index=selected_feature, 
                        columns='Predicted_Class', 
                        values='Confidence', 
                        fill_value=0
                    )
                    st.bar_chart(chart_data)
        
        with tabs[1]:
            st.write("Explore patterns in your data:")
            
            if len(st.session_state.features) >= 2:
                feature1 = st.selectbox("X-axis feature:", [f['name'] for f in st.session_state.features])
                feature2 = st.selectbox("Y-axis feature:", [f['name'] for f in st.session_state.features], index=1)
                
                if feature1 != feature2:
                    st.write(f"**{feature1} vs {feature2}**")
                    
                    # Create scatter plot with matplotlib
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Get unique target values and assign colours
                    unique_targets = st.session_state.data[st.session_state.target_variable].unique()
                    colours = plt.cm.Set1(np.linspace(0, 1, len(unique_targets)))
                    
                    for i, target in enumerate(unique_targets):
                        mask = st.session_state.data[st.session_state.target_variable] == target
                        ax.scatter(
                            st.session_state.data[mask][feature1], 
                            st.session_state.data[mask][feature2],
                            c=[colours[i]], 
                            label=target,
                            alpha=0.7
                        )
                    
                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    ax.legend()
                    ax.set_title(f"{feature1} vs {feature2}")
                    st.pyplot(fig)
                    plt.close()
        
        with tabs[2]:
            st.write("🧪 **Try these experiments:**")
            
            experiments = [
                "Add more data points and see if the tree structure changes",
                "Try different tree parameters (depth, split criterion) and compare results",
                "Look for patterns: which features seem most important?",
                "Test edge cases: what happens with unusual combinations?",
                "Check if the model makes sense: do the predictions align with your intuition?"
            ]
            
            for i, exp in enumerate(experiments, 1):
                st.write(f"{i}. {exp}")

# Educational sidebar
with st.sidebar.expander("📚 Learn More", expanded=False):
    st.write("""
    **Decision Tree Concepts:**
    
    🌟 **Features** - The input characteristics (age, preferences, etc.)
    
    🎯 **Target Variable** - What you're trying to predict
    
    🌳 **Tree Depth** - How many questions the algorithm can ask
    
    📊 **Feature Importance** - Which characteristics matter most
    
    🔍 **Entropy/Gini** - Different ways to measure "information gain"
    """)