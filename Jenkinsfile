pipeline {
    agent any

    stages {
        stage('Clone repository') {
            steps {
                script {
                    echo 'Cloning the Repository to our workspace'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops_git_token', url: 'https://github.com/UsmanKhan555/MLOpsProject.git']])
                }
            }
        }

        stage('Install dependencies') {
            steps {
                script {
                    echo 'Installing dependencies'
                    // Install dependencies, including pytest
                    sh "python3 -m pip install --upgrade pip"
                    sh "python3 -m pip install --break-system-packages -r requirements.txt"
                }
            }
        }

        stage('Test the code') {
            steps {
                script {
                    echo 'Running tests with pytest'
                    // Run tests using pytest
                    sh "pytest test_app.py"
                }
            }
        }
    }

}