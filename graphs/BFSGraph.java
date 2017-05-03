/**
 * Created by alokpathy on 12/19/15.
 */
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;

public class BFSGraph {

    private TreeMap<Integer, TreeSet<Integer>> adjList;

    public BFSGraph(String path, int currRoot, int u, int v, boolean insertion) throws IOException {
        adjList = new TreeMap<Integer, TreeSet<Integer>>();

        BufferedReader in = new BufferedReader(new FileReader(path));
        StringTokenizer tokenizer = new StringTokenizer(in.readLine());

        int NV = Integer.parseInt(tokenizer.nextToken());
        int NE = Integer.parseInt(tokenizer.nextToken());

        String line = "";
        int vertex = 1;
        while ((line = in.readLine()) != null) {
            String[] neighbors = line.split(" ");
            for (int i = 0; i < neighbors.length; i++) {
                if (neighbors[i].equals("")) {
                    continue;
                }
                if (!adjList.containsKey(vertex)) {
                    TreeSet<Integer> list = new TreeSet<Integer>();
                    list.add(Integer.parseInt(neighbors[i]));
                    adjList.put(vertex, list);
                } else {
                    TreeSet<Integer> list = adjList.get(vertex);
                    list.add(Integer.parseInt(neighbors[i]));
                    adjList.put(vertex, list);
                }
            }
            vertex++;
        }

        if (insertion) {
            if (!adjList.containsKey(u)) {
                TreeSet<Integer> list = new TreeSet<Integer>();
                list.add(v);
                adjList.put(u, list);
            } else {
                TreeSet<Integer> list = adjList.get(u);
                list.add(v);
                adjList.put(u, list);
            }

            if (!adjList.containsKey(v)) {
                TreeSet<Integer> list = new TreeSet<Integer>();
                list.add(u);
                adjList.put(v, list);
            } else {
                TreeSet<Integer> list = adjList.get(v);
                list.add(u);
                adjList.put(v, list);
            }
        } else {
            System.out.println("Deletion not implemented yet.");
            return;
        }
        bfs(currRoot, NV);
    }

    private void bfs(int currRoot, int NV) {
        int[] dist = new int[NV + 1];
        boolean[] visited = new boolean[NV + 1];
        Queue<Integer> queue = new LinkedList<Integer>();
        queue.add(currRoot);

        for (int i = 0; i < dist.length; i++) {
            dist[i] = Integer.MAX_VALUE;
        }
        dist[currRoot] = 0;

        while (!queue.isEmpty()) {
            int u = queue.poll();
            visited[u] = true;
            TreeSet<Integer> neighbors = adjList.get(u);
            for (int v : neighbors) {
                if (!visited[v]) {
                    queue.add(v);
                    dist[v] = Math.min(dist[v], dist[u] + 1);
                }
            }
        }

        for (int i = 1; i < dist.length; i++) {
            System.out.println("Vertex: " + i + " Dist: " + dist[i]);
        }
    }

    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Usage: java BFSGraph <path> <root> <u> <v>");
            return;
        }
        String path = args[0];
        int currRoot = Integer.parseInt(args[1]);
        int u = Integer.parseInt(args[2]);
        int v = Integer.parseInt(args[3]);
        try {
            new BFSGraph(path, currRoot, u, v, true);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
