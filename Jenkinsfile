pipeline {
    agent any

    stages {
        stage('Clone repository') {
            steps {
                script {
                    // Cloning the Repository to our workspace
                    echo 'Cloning the Repository to our workspace'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops-git-token', url: 'https://github.com/UsmanKhan555/NumberML.git']])
                }
            }
        }

        stage('Test'){
            steps {
                script {
                    echo 'Testing the code'
                    sh "python3 --version"
                }
            }
        }


    }
}
