pipeline {
    agent any

    stages {
        stage('Clone repository') {
            steps {
                script {
                    // Cloning the Repository to our workspace
                    echo 'Cloning the Repository to our workspace'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops_git_token', url: 'https://github.com/UsmanKhan555/MLOpsProject.git']])
                }
            }
        }

        stage('Check out the code') {
            steps {
                script {
                    echo 'Checking out the code'
                    sh "python -m pip install --break-system-packages -r requirements.txt"
                }
            }
        }

        stage('Test the code') {
            steps {
                script {
                    echo 'Testing the code'
                    sh "pytest test_app.py" 
                }
            }
        }
    }
}
