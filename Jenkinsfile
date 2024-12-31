pipeline {
    agent any
    environment {
        DOCKER_IMAGE = "mlopsproject"
    }
    stages {
        stage('Clone repository') {
            steps {
                script {
                    echo 'Cloning the Repository to our workspace'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops_git_token', url: 'https://github.com/UsmanKhan555/MLOpsProject.git']])
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    sh "python -m pip install --upgrade pip --break-system-packages"
                    sh "python -m pip install --break-system-packages -r requirements.txt"
                }
            }
        }

        stage('Test') {
            steps {
                script {
                    sh "python -m pytest test_app.py"
                }
            }
        }
        stage('Trivy Scan') {
            steps {
                script {
                    echo "Running Trivy Scan"
                    sh "trivy fs --format table -o trivy-fs-report.html ."
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker Image"
                    sh "docker build -t ${DOCKER_IMAGE} ."
                }
            }
        }
    }
}