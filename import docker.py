import docker
client = docker.from_env()

client.containers.run("ubuntu:latest",
                      "echo hello not world",
                      "echo test")
'hello not world\n'
'test\n'

# client.images.pull('nginx')

#Container = client.containers.get('9d07d8bac3bc5e4ce04571fd91a841e85a7d219487047d1f36064c2a0cfc77bc')

#client.containers.run("bfirsh/reticulate-splines", detach=True, "echo test")
#'test\n'
#Container '9d07d8bac3bc5e4ce04571fd91a841e85a7d219487047d1f36064c2a0cfc77bc'

#client.containers.run("bfirsh/reticulate-splines", detach=True)
#<Container '6a8145d35df69026076cef534b4387ee5207571f490c7f6a22264373d0bbf1e9'>

#client.containers.list()
#Container <'2ddb0a73768b95fe17baa210f05093c2cfef539714ddb01e877f4915bc54929b'>

#container = client.containers.get('45e6d2de7c54')

#container.attrs['Config']['Image']
#"bfirsh/reticulate-splines"

#container.logs()
#"Reticulating spline 1...\n"

#container.stop()

#for line in container.logs(stream=True):
#print(line.strip())
#Reticulating spline 2...
#Reticulating spline 3...
#

#client.images.pull('nginx')
#<Image 'nginx'>

#client.images.list()
#[<Image 'ubuntu'>, <Image 'nginx'>, ...]