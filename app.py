from flask import Flask, render_template, request, jsonify
from threading import Thread
import queue
import time
from utils import doc_manager, agent_manager

# Initialize Flask app
app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')


app.config['TEMPLATES_AUTO_RELOAD'] = True


def format_thought_process(intermediate_steps):
    formatted_steps = []
    for step in intermediate_steps:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            thought = action.log.split('Thought:')[1].split('Action:')[0].strip() if 'Thought:' in action.log else ''
            formatted_steps.append({
                'thought': thought,
                'action': action.tool,
                'action_input': action.tool_input,
                'observation': str(observation)
            })
        else:
            formatted_steps.append({
                'thought': 'Error: Unexpected step format',
                'action': 'Unknown',
                'action_input': 'Unknown',
                'observation': 'Unknown'
            })
    return formatted_steps

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_docs', methods=['POST'])
def update_docs():
    try:
        num_docs = doc_manager.update_documentation()
        # Reinitialize agent with new documentation
        global agent_manager
        agent_manager = AgentManager(doc_manager)
        return jsonify({
            'status': 'success',
            'message': f'Successfully updated documentation with {num_docs} documents'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        result = agent_manager.agent_executor.invoke({"input": question})
        
        # Check if the result contains the expected keys
        if 'output' not in result:
            raise KeyError("The 'output' key is missing from the result")
        
        # Format the response
        response_data = {
            'answer': result['output'],
            'steps': format_thought_process(result.get('intermediate_steps', []))
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing question: {str(e)}")  # For debugging
        return jsonify({
            'error': 'An error occurred while processing your question',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)