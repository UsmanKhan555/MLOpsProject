pipeline {
    agent any
    stages {
        stage('Clone repository') {
            steps {
                script {
                    echo 'Cloning the Repository to our workspace'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops-git-token', url: 'https://github.com/UsmanKhan555/MLOpsProject.git']])
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

        // stage('Docker Build') {
        //     steps {
        //         script {
        //             echo 'Building Docker image'
        //             sh 'docker build -t my-docker-image:latest .'
        //         }
        //     }
        // }

    }
}